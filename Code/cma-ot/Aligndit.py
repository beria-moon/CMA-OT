from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama import LlamaConfig
from geomloss import SamplesLoss
from .loss import FGW_OT_Loss
from .modules import (
    TimestepEmbedding,
    ConvPositionEmbedding,
    AdaLayerNormZero_Final,
    _prepare_decoder_attention_mask,
)


def log_sinkhorn_knopp_batch(log_K, a, b, n_iters=100, tol=1e-9):
    """
    批量 log-domain Sinkhorn-Knopp
    log_K: (B, N, M) 对应 -G/eps
    a: (B, N)
    b: (B, M)
    """
    B, N, M = log_K.shape
    log_u = torch.zeros(B, N, device=log_K.device)  # log(u)
    log_v = torch.zeros(B, M, device=log_K.device)  # log(v)

    log_a = torch.log(a + 1e-16)
    log_b = torch.log(b + 1e-16)

    for _ in range(n_iters):
        log_u_prev = log_u.clone()

        # 更新 log(u)
        # log_u = log_a - logsumexp(log_K + log_v, axis=2)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)

        # 更新 log(v)
        # log_v = log_b - logsumexp(log_K^T + log_u, axis=1)
        log_v = log_b - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

        # 收敛性检查
        if torch.max(torch.abs(log_u - log_u_prev)) < tol:
            break

    # 返回最终 pi
    log_pi = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # (B, N, M)
    pi = torch.exp(log_pi)
    return pi


def gw_ot_loss(
    expert_repr, gen_repr,
    outer_iters=20, sinkhorn_iters=100, eps=1e-3,
    verbose=False
):
    """
    Log-domain 数值稳定版 Gromov-Wasserstein OT 损失
    expert_repr: (B, T1, D)
    gen_repr: (B, T2, D)
    """
    B, T1, D = expert_repr.shape
    T2 = gen_repr.shape[1]
    device = expert_repr.device

    # 1. 特征归一化
    expert_repr = F.normalize(expert_repr, p=2, dim=-1)
    gen_repr = F.normalize(gen_repr, p=2, dim=-1)

    # 2. 计算 pairwise 距离矩阵 (平方 L2)
    Cx = torch.cdist(expert_repr, expert_repr, p=2) ** 2
    Cy = torch.cdist(gen_repr, gen_repr, p=2) ** 2

    # 归一化到 [0,1]，避免溢出
    Cx = Cx / (Cx.amax(dim=(1,2), keepdim=True) + 1e-16)
    Cy = Cy / (Cy.amax(dim=(1,2), keepdim=True) + 1e-16)

    # 3. 初始分布
    a = torch.ones(B, T1, device=device) / T1
    b = torch.ones(B, T2, device=device) / T2
    pi = torch.ones(B, T1, T2, device=device) / (T1 * T2)

    # 4. 迭代 GW OT
    for it in range(outer_iters):
        # 边缘分布
        p = pi.sum(dim=2).clamp(min=1e-16)
        q = pi.sum(dim=1).clamp(min=1e-16)

        # G 计算
        A = torch.bmm(Cx, p.unsqueeze(-1)).squeeze(-1)     # (B,T1)
        B_vec = torch.bmm(Cy, q.unsqueeze(-1)).squeeze(-1) # (B,T2)
        V = torch.bmm(pi, Cy.transpose(1, 2))              # (B,T1,T2)
        U = torch.bmm(Cx, V)                               # (B,T1,T2)
        G = A.unsqueeze(2) + B_vec.unsqueeze(1) - 2.0 * U  # (B,T1,T2)

        # 归一化 G 避免溢出
        G_max = G.amax(dim=(1,2), keepdim=True)
        G = G / (G_max + 1e-16)

        # log-domain kernel
        log_K = -G / eps

        # log-Sinkhorn
        pi = log_sinkhorn_knopp_batch(log_K, a, b, n_iters=sinkhorn_iters)

        if verbose and (it % 5 == 0):
            approx_loss = (G * pi).sum(dim=(1,2)).mean()
            print(f"[iter {it}] GW approx: {approx_loss.item():.6f}")

    # 5. 最终 GW loss
    p = pi.sum(dim=2)
    q = pi.sum(dim=1)
    term1 = (Cx * torch.bmm(p.unsqueeze(2), p.unsqueeze(1))).sum(dim=(1,2))
    term2 = (Cy * torch.bmm(q.unsqueeze(2), q.unsqueeze(1))).sum(dim=(1,2))
    term3 = -2.0 * ((torch.bmm(torch.bmm(Cx, pi), Cy.transpose(1, 2)) * pi).sum(dim=(1,2)))
    loss = (term1 + term2 + term3).mean()

    return loss, pi




class FeatureAlign(nn.Module):
    """
    将 [B, T, D] 特征对齐到相同的时间步和维度
    - T 通过 Adaptive Pooling 对齐
    - D 通过 Linear 映射到统一维度
    """
    def __init__(self, in_dim, out_dim=512, target_len=108):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.pool = nn.AdaptiveAvgPool1d(target_len)  # 对齐时间步
        self.out_dim = out_dim
        self.target_len = target_len

    def forward(self, x):
        """
        输入: x [B, T, D]
        输出: [B, target_len, out_dim]
        """
        # 投影特征维度
        x = self.proj(x)  # [B, T, out_dim]

        # 转换为 [B, out_dim, T]，方便池化
        x = x.transpose(1, 2)  # [B, out_dim, T]

        # 池化到目标长度
        x = self.pool(x)  # [B, out_dim, target_len]

        # 转回 [B, target_len, out_dim]
        x = x.transpose(1, 2)
        return x

# -------------------------
# 简化的 REPA 损失
# -------------------------
class SimpleREPALoss(nn.Module):
    def __init__(self, dit_dim, jukebox_dim, hidden_dim=512, target_len=108):
        super().__init__()
        self.dit_align = FeatureAlign(dit_dim, out_dim=hidden_dim, target_len=target_len)
        self.jukebox_align = FeatureAlign(jukebox_dim, out_dim=hidden_dim, target_len=target_len)

    def forward(self, dit_features, jukebox_features):
        """
        dit_features: [B, T1, D1]
        jukebox_features: [B, T2, D2]
        """
        dit_proj = self.dit_align(dit_features)      # [B, target_len, H]
        juke_proj = self.jukebox_align(jukebox_features)  # [B, target_len, H]

        # L2 归一化
        dit_proj = F.normalize(dit_proj, p=2, dim=-1)
        juke_proj = F.normalize(juke_proj, p=2, dim=-1)

        # 逐时间步余弦相似度
        cosine_sim = (dit_proj * juke_proj).sum(dim=-1)  # [B, target_len]
        # print("cosine_sim:",cosine_sim.shape)

        # 损失 = 1 - 平均余弦相似度
        loss = 1 - cosine_sim.mean()
        return loss
# -------------------------
# OT REPA 损失
# -------------------------
class OTGeomLoss(nn.Module):
    def __init__(self, dit_dim, jukebox_dim, proj_dim=512, p=2, blur=0.05):
        """
        dit_dim: DiT 特征维度
        jukebox_dim: Jukebox 特征维度
        proj_dim: 投影到相同维度
        p: Sinkhorn 距离的指数
        blur: Sinkhorn 模糊参数
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.dit_proj = nn.Linear(dit_dim, proj_dim)
        self.juke_proj = nn.Linear(jukebox_dim, proj_dim)
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    def forward(self, dit_features, jukebox_features):
        """
        dit_features: [B, T1, D1]
        jukebox_features: [B, T2, D2]
        """
        # 投影到相同维度
        dit_aligned = self.dit_proj(dit_features)    # [B, T1, proj_dim]
        juke_aligned = self.juke_proj(jukebox_features)  # [B, T2, proj_dim]

        # # L2 归一化
        # dit_aligned = F.normalize(dit_aligned, p=2, dim=-1)
        # juke_aligned = F.normalize(juke_aligned, p=2, dim=-1)

        # 计算 OT Sinkhorn loss
        loss = self.loss_fn(dit_aligned, juke_aligned)
        return loss
# -------------------------
# 输入嵌入
# -------------------------
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, dance_dim, out_dim, cond_dim, style_dim):
        super().__init__()
        # + time_emb
        self.proj = nn.Linear(mel_dim*2 + dance_dim + cond_dim + style_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, dance, style, time_emb, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        time_emb = time_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        # print(x.shape,cond.shape,dance.shape,style.shape,time_emb.shape)
        input_tensor = torch.cat((x, cond, dance, style, time_emb), dim=-1)
        x = self.proj(input_tensor)
        x = self.conv_pos_embed(x) + x
        return x


# -------------------------
# AlignDiT 主体
# -------------------------
class AlignDiT(nn.Module):
    def __init__(
        self,
        dim,
        depth=12,
        heads=12,
        dim_head=64,
        dropout=0.1,
        ff_mult=6,
        mel_dim=64,
        text_dim=512,
        style_dim=512,
        conv_layers=0,
        long_skip_connection=False,
        max_frames=2048,
        use_multilayer_cma_ot=False,
        adaptive_alpha=True,
    ):
        super().__init__()

        self.max_frames = max_frames
        self.use_multilayer_cma_ot = use_multilayer_cma_ot
        self.adaptive_alpha = adaptive_alpha
        self.last_repa_stats = {}
        cond_dim = 512

        self.time_embed = TimestepEmbedding(cond_dim)
        self.start_time_embed = TimestepEmbedding(cond_dim)

        self.input_embed = InputEmbedding(
            mel_dim=mel_dim,
            dance_dim=text_dim,
            out_dim=dim,
            cond_dim=cond_dim,
            style_dim=style_dim
        )

        self.dim = dim
        self.depth = depth

        if dim != heads * dim_head:
            raise ValueError(
                f"AlignDiT expects dim == heads * dim_head, got dim={dim}, "
                f"heads={heads}, dim_head={dim_head}"
            )

        llama_config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=dim * ff_mult,
            hidden_act="silu",
            max_position_embeddings=self.max_frames,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            head_dim=dim_head,
        )
        llama_config._attn_implementation = "sdpa"

        self.transformer_blocks = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNormZero_Final(dim, cond_dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        def _make_mapper(out_dim):
            return nn.Sequential(
                nn.Linear(dim, 256),
                nn.GELU(),
                nn.Linear(256, out_dim),
            )

        fgw_kwargs = dict(
            outer_iters=5,
            sinkhorn_iters=50,
            eps=1e-2,
            feature_loss="cosine",
            adaptive_alpha=adaptive_alpha,
        )

        if use_multilayer_cma_ot:
            # DiT layer index (0-based) -> jukebox level; paper: L2/L6/L10
            self.cma_ot_layers = {
                1: "bottom",
                5: "middle",
                9: "top",
            }
            self.extract_layers = list(self.cma_ot_layers.keys())
            self.dit_mappers = nn.ModuleDict(
                {level: _make_mapper(64) for level in ["bottom", "middle", "top"]}
            )
            self.repa_losses = nn.ModuleDict(
                {
                    level: FGW_OT_Loss(expert_dim=64, gen_dim=64, alpha=0.5, **fgw_kwargs)
                    for level in ["bottom", "middle", "top"]
                }
            )
            self.repa_loss_1 = None
        else:
            self.cma_ot_layers = {8: "mert"}
            self.extract_layers = [8]
            self.dit_mapper_1 = _make_mapper(1024)
            self.repa_loss_1 = FGW_OT_Loss(
                expert_dim=1024,
                gen_dim=1024,
                alpha=0.5,
                adaptive_alpha=False,
                **{k: v for k, v in fgw_kwargs.items() if k != "adaptive_alpha"},
            )
            self.dit_mappers = None
            self.repa_losses = None

    def forward(
        self,
        x: torch.Tensor,        # noised input audio [b, n, mel_dim]
        cond: torch.Tensor,     # masked cond audio [b, n, mel_dim]
        text: torch.Tensor,     # 舞蹈特征 [b, n, text_dim]
        time: torch.Tensor,     # time step [b]
        drop_audio_cond=False,
        style_prompt=None,
        start_time=None,
        jukebox_features=None,
        curriculum_weights: dict[str, float] | None = None,
        return_intermediate=True,
        drop_text=False,
        drop_prompt=False
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # 时间嵌入
        t = self.time_embed(time)
        if start_time is None:
            start_time = torch.zeros(batch, device=x.device)
        s_t = self.start_time_embed(start_time)
        c = t + s_t

        style_embed = style_prompt
        if drop_text:
            text = torch.zeros_like(text)
        if drop_prompt:
            style_embed = torch.zeros_like(style_embed)
        x = self.input_embed(x, cond, text, style_embed, c, drop_audio_cond=drop_audio_cond)

        if self.long_skip_connection is not None:
            residual = x

        # 注意力 mask
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch, 1)
        rotary_embed = self.rotary_emb(x, pos_ids)  # TODO: 确认接口是否对
        attention_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=x.device)
        attention_mask = _prepare_decoder_attention_mask(attention_mask, (batch, seq_len), x)

        # 中间特征存储
        intermediate_features = {}

        dit_features = []
        for i, block in enumerate(self.transformer_blocks):
            x, *_ = block(x, attention_mask=attention_mask, position_embeddings=rotary_embed)
            if return_intermediate and i in self.extract_layers:
                intermediate_features[i] = x.clone()
            dit_features.append(x.detach())
        
        # print(jukebox_features.shape)
        # exit()
        # 计算 REPA / CMA-OT 损失
        repa_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        self.last_repa_stats = {}
        if jukebox_features is not None:
            if self.use_multilayer_cma_ot:
                repa_loss, self.last_repa_stats = self._multilayer_cma_ot_loss(
                    intermediate_features, jukebox_features, curriculum_weights
                )
            else:
                mapped_dit = self.dit_mapper_1(intermediate_features[8])
                repa_loss, _ = self.repa_loss_1(mapped_dit, jukebox_features)
            
            # if jukebox_features.shape[1]==l1:
            #     jukebox_top = jukebox_features[:,:l1,:]
            #     mapped_dit_1  = self.dit_mapper_1(intermediate_features[1])
            #     repa_loss_t,_ = self.repa_loss_1(mapped_dit_1,jukebox_top)
            #     repa_loss_m = torch.tensor(0.0, device=repa_loss_t.device)
            #     repa_loss_b = torch.tensor(0.0, device=repa_loss_t.device)
            # elif jukebox_features.shape[1]==l1+l2:
            #     jukebox_top = jukebox_features[:,:l1,:]
            #     mapped_dit_1  = self.dit_mapper_1(intermediate_features[1])
            #     repa_loss_t,_ = self.repa_loss_1(mapped_dit_1,jukebox_top)

            #     jukebox_mid = jukebox_features[:,l1:l1+l2,:]
            #     mapped_dit_2 = self.dit_mapper_2(intermediate_features[5])
            #     repa_loss_m,_ = self.repa_loss_2(mapped_dit_2,jukebox_mid)
            #     repa_loss_b = torch.tensor(0.0, device=repa_loss_t.device)
            # else:
            #     jukebox_top = jukebox_features[:,:l1,:]
            #     mapped_dit_1  = self.dit_mapper_1(intermediate_features[1])
            #     repa_loss_t,_ = self.repa_loss_1(mapped_dit_1,jukebox_top)

            #     jukebox_mid = jukebox_features[:,l1:l1+l2,:]
            #     mapped_dit_2 = self.dit_mapper_2(intermediate_features[5])
            #     repa_loss_m,_ = self.repa_loss_2(mapped_dit_2,jukebox_mid)

            #     jukebox_bot = jukebox_features[:,l1+l2:,:]
            #     mapped_dit_3 = self.dit_mapper_3(intermediate_features[9])
            #     repa_loss_b,_ = self.repa_loss_3(mapped_dit_3,jukebox_bot)


            # if jukebox_features.shape[1]==l1:
            #     jukebox_top = jukebox_features[:,:l1,:]
            #     mapped_dit_1  = self.dit_mapper_1(intermediate_features[5])
            #     repa_loss,_ = self.repa_loss_1(mapped_dit_1,jukebox_top)
            #     # repa_loss_m = torch.tensor(0.0, device=repa_loss_t.device)
            #     # repa_loss_b = torch.tensor(0.0, device=repa_loss_t.device)
            # elif jukebox_features.shape[1]==l2:

            #     jukebox_mid = jukebox_features[:,l1:l1+l2,:]
            #     mapped_dit_2 = self.dit_mapper_1(intermediate_features[5])
            #     repa_loss,_ = self.repa_loss_1(mapped_dit_2,jukebox_mid)
            #     # repa_loss_b = torch.tensor(0.0, device=repa_loss_t.device)
            # else:

            #     jukebox_bot = jukebox_features[:,l1+l2:,:]
            #     mapped_dit_3 = self.dit_mapper_1(intermediate_features[9])
            #     repa_loss,_ = self.repa_loss_1(mapped_dit_3,jukebox_bot)



            

            # jukebox_top = jukebox_features[:,:1722,:]
            # jukebox_mid = jukebox_features[:,1722:1722+6890,:]
            # jukebox_bot = jukebox_features[:,1722+6890:,:]
            # print(jukebox_top.shape,jukebox_mid.shape,jukebox_bot.shape)
            # exit()

            # 使用映射网络将 jukebox 特征投射到 DiT 空间 [B, T, 768]
            # mapped_dit  = self.dit_mapper_1(intermediate_features[8])
            # mapped_dit_1 = self.dit_mapper_1(intermediate_features[9])
            # mapped_dit_2 = self.dit_mapper_2(intermediate_features[5])
            # mapped_dit_3 = self.dit_mapper_3(intermediate_features[1])

            # mapped_jukebox = self.jukebox_mapper(jukebox_features)
            # print(intermediate_features[8].shape,jukebox_features.shape)
            # repa_loss,_ = self.repa_loss_1(mapped_dit_1, mert_features)
            # repa_loss_m,_ = self.repa_loss_2(mapped_dit_2,jukebox_mid)
            # repa_loss_b,_ = self.repa_loss_3(mapped_dit_3,jukebox_bot)

            # repa_loss = repa_loss_1+repa_loss_2+repa_loss_3
            # print(repa_loss)
            # repa_loss,_ = gw_ot_loss(intermediate_features[7],jukebox_features)
          
            # repa_loss,_ = self.repa_loss_1(mapped_dit,jukebox_features)
            # print('repa_loss:',repa_loss)
            # exit()

        # skip connection
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, c)
        output = self.proj_out(x)

        # return output,repa_loss_t,repa_loss_m,repa_loss_b
        return output, repa_loss

    @staticmethod
    def _to_time_major(feat):
        """(B, D, T) or (B, T, D) -> (B, T, D)."""
        if feat.ndim != 3:
            raise ValueError(f"Expected 3D feature, got {feat.shape}")
        if feat.shape[1] == 64 and feat.shape[2] != 64:
            return feat.permute(0, 2, 1).contiguous()
        return feat

    def _multilayer_cma_ot_loss(self, intermediate_features, jukebox_features, curriculum_weights=None):
        if not isinstance(jukebox_features, dict):
            raise TypeError(
                "Multilayer CMA-OT expects jukebox_features dict with keys "
                "bottom/middle/top."
            )

        total = torch.zeros((), device=next(iter(intermediate_features.values())).device)
        stats = {}
        for layer_idx, level in self.cma_ot_layers.items():
            expert = self._to_time_major(jukebox_features[level])
            mapped = self.dit_mappers[level](intermediate_features[layer_idx])
            loss_k, _, layer_stats = self.repa_losses[level](
                mapped, expert, return_stats=True
            )
            weight = 1.0 if curriculum_weights is None else float(curriculum_weights.get(level, 0.0))
            total = total + weight * loss_k
            paper_layer = layer_idx + 1
            stats[f"layer_{paper_layer}_{level}"] = {
                "alpha_k_mean": layer_stats["alpha_k_mean"],
                "m_k_mean": layer_stats["m_k_mean"],
                "n_k_mean": layer_stats["n_k_mean"],
                "alpha_k": layer_stats["alpha_k"],
            }
        stats["alpha_k_mean"] = float(
            sum(stats[k]["alpha_k_mean"] for k in stats if k.startswith("layer_"))
            / max(len(self.cma_ot_layers), 1)
        )
        return total, stats
  