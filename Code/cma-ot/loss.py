import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlign(nn.Module):
    """
    将 [B, T, D] 特征对齐到相同的时间步和维度
    - T 通过自定义 Adaptive Pooling 对齐
    - D 通过 Linear 映射到统一维度
    """
    def __init__(self, in_dim, out_dim=512, target_len=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim
        self.target_len = target_len

    def forward(self, x, pool=True):
        """
        输入: x [B, T, D]
        pool=True:  [B, target_len, out_dim]
        pool=False: [B, T, out_dim]
        """
        x = self.proj(x)
        if not pool:
            return x
        x = x.transpose(1, 2)
        x = self.adaptive_avg_pool_1d(x, self.target_len)
        x = x.transpose(1, 2)
        return x

    @staticmethod
    def adaptive_avg_pool_1d(x, target_len):
        """
        自定义 1D 平均池化，避免 torch.nn.AdaptiveAvgPool1d 的 CUDA shared memory bug
        输入: x [B, C, T]
        输出: [B, C, target_len]
        """
        B, C, T = x.shape
        if T == target_len:
            return x
        ratio = T // target_len
        x = x[:, :, :ratio * target_len]       # 截断能整除的部分
        x = x.view(B, C, target_len, ratio)    # 分段
        x = x.mean(dim=-1)                     # 对每段求均值
        return x
class FGW_OT_Loss(nn.Module):
    def __init__(
        self,
        expert_dim,
        gen_dim,
        alpha=0.5,
        outer_iters=5,
        sinkhorn_iters=50,
        eps=1e-2,
        feature_loss="cosine",
        adaptive_alpha=False,
        alpha_eps=1e-6,
    ):
        """
        FGW OT Loss (batch-friendly, Tx≠Ty, feature projection)
        expert_dim: D_expert
        gen_dim:    D_gen
        adaptive_alpha: if True, alpha_k = sigmoid(log(n_k / (m_k + eps)))
        """
        super().__init__()
        self.alpha = alpha
        self.outer_iters = outer_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.feature_loss = feature_loss
        self.adaptive_alpha = adaptive_alpha
        self.alpha_eps = alpha_eps

        self.dit_align = FeatureAlign(in_dim=gen_dim, out_dim=512, target_len=108)
        self.jukebox_align = FeatureAlign(in_dim=expert_dim, out_dim=512, target_len=108)


    def log_sinkhorn(self, log_K, a, b, n_iters=50, tol=1e-9):
        N, M = log_K.shape
        f = torch.zeros(N, device=log_K.device)
        g = torch.zeros(M, device=log_K.device)
        log_a = torch.log(a + 1e-20)
        log_b = torch.log(b + 1e-20)
        for _ in range(n_iters):
            f_prev = f.clone()
            f = log_a - torch.logsumexp(log_K + g[None, :], dim=1)
            g = log_b - torch.logsumexp(log_K.T + f[None, :], dim=1)
            if torch.max(torch.abs(f - f_prev)) < tol:
                break
        log_pi = log_K + f[:, None] + g[None, :]
        pi = torch.exp(torch.clamp(log_pi, min=-50, max=50))  # 防止exp溢出
        pi = pi / (pi.sum() + 1e-9)
        return pi
    
    def pairwise_squared_distances(self, X, Y):
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        dist = X_norm + Y_norm.t() - 2*X@Y.t()

        return torch.clamp(dist, min=0.0,max=10.0)

    def forward(self, gen_repr, expert_repr, verbose=False, return_stats=False):
        """
        expert_repr: (B, Tx, D_expert)
        gen_repr:    (B, Ty, D_gen)
        """
        B = expert_repr.shape[0]
        device = expert_repr.device

        dit_mapped = F.normalize(self.dit_align(gen_repr, pool=False), p=2, dim=-1)
        juke_mapped = F.normalize(self.jukebox_align(expert_repr, pool=False), p=2, dim=-1)

        total_loss = 0.0
        pis = []
        stats = {"alpha_k": [], "m_k": [], "n_k": []}

        for b in range(B):
            X = expert_repr[b]
            Y = gen_repr[b]
            N, M = X.shape[0], Y.shape[0]

            Cx = self.pairwise_squared_distances(X, X)
            Cy = self.pairwise_squared_distances(Y, Y)
            Cx = Cx / (Cx.max() + 1e-16)
            Cy = Cy / (Cy.max() + 1e-16)

            a = torch.ones(N, device=device) / N
            b_vec = torch.ones(M, device=device) / M
            pi = torch.ones(N, M, device=device) / (N * M)

            for it in range(self.outer_iters):
                G = torch.einsum("ik,kl,jl->ij", Cx, pi, Cy)
                log_K = -G / self.eps
                pi = self.log_sinkhorn(log_K, a, b_vec, n_iters=self.sinkhorn_iters)
                if verbose and it % 5 == 0:
                    print(f"[batch {b} iter {it}] GW approx: {torch.sum(G * pi).item():.6f}")

            p = pi.sum(dim=1)
            q = pi.sum(dim=0)
            term1 = torch.einsum("ij,i,j->", Cx, p, p)
            term2 = torch.einsum("ij,i,j->", Cy, q, q)
            tmp = torch.einsum("ik,kl,jl->ij", Cx, pi, Cy)
            term3 = torch.sum(tmp * pi)
            n_k = term1 + term2 - 2 * term3

            X_feat = juke_mapped[b]
            Y_feat = dit_mapped[b]
            if self.feature_loss == "mse":
                feat_cost = self.pairwise_squared_distances(X_feat, Y_feat)
            elif self.feature_loss == "cosine":
                feat_cost = 1.0 - X_feat @ Y_feat.t()
            else:
                raise ValueError("feature_loss must be 'mse' or 'cosine'")
            feat_cost = feat_cost / (feat_cost.max() + 1e-16)
            m_k = (feat_cost * pi).sum() / (pi.sum() + 1e-16)

            if self.adaptive_alpha:
                alpha_k = torch.sigmoid(torch.log(n_k / (m_k + self.alpha_eps)))
            else:
                alpha_k = torch.tensor(self.alpha, device=device, dtype=m_k.dtype)

            loss = (1.0 - alpha_k) * m_k + alpha_k * n_k

            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"Warning: NaN/Inf in batch {b}, fallback to feature_loss")
                loss = m_k
                alpha_k = torch.tensor(0.0, device=device, dtype=m_k.dtype)

            total_loss += loss
            pis.append(pi.detach())
            if return_stats:
                stats["alpha_k"].append(float(alpha_k.detach().cpu()))
                stats["m_k"].append(float(m_k.detach().cpu()))
                stats["n_k"].append(float(n_k.detach().cpu()))

        total_loss /= B
        if return_stats:
            for key in ("alpha_k", "m_k", "n_k"):
                stats[key + "_mean"] = float(sum(stats[key]) / max(len(stats[key]), 1))
            return total_loss, pis, stats
        return total_loss, pis
