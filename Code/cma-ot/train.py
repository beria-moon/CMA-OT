from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import build_model, load_config, seed_everything
from .dataset import CMAOTDataset, collate_cma_ot
from .schedule import curriculum_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMA-OT dance-to-music model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["train"]["seed"])
    accelerator = Accelerator(gradient_accumulation_steps=config["train"]["grad_accumulation_steps"])
    dataset = CMAOTDataset(args.train_manifest, require_teacher=True)
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=config["train"]["num_workers"], collate_fn=collate_cma_ot)
    model = build_model(config)
    optimizer = Adam(model.parameters(), lr=config["train"]["learning_rate"], betas=(0.9, 0.95))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, config["train"]["epochs"]):
        model.train()
        weights = curriculum_weights(epoch)
        for batch in loader:
            teachers = {level: batch[f"jukebox_{level}"] for level in ("bottom", "middle", "top")}
            with accelerator.accumulate(model):
                loss, flow_loss, alignment_loss = model(
                    batch["latent"], batch["pose"], style_prompt=batch["style_i3d"],
                    jukebox_features=teachers, curriculum_weights=weights,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["train"]["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process:
                accelerator.print(f"epoch={epoch + 1} loss={loss.item():.5f} flow={flow_loss.item():.5f} align={alignment_loss.item():.5f} weights={weights}")
        if accelerator.is_main_process:
            torch.save({"epoch": epoch, "model": accelerator.get_state_dict(model), "optimizer": optimizer.state_dict(), "config": config}, output_dir / "last.pt")


if __name__ == "__main__":
    main()
