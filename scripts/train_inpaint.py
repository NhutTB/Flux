r"""
Training script for FLUX.2 inpainting fine-tuning.

This script fine-tunes the FLUX.2-klein-4B model to support mask-conditioned
inpainting using the Channel Concatenation approach. Only the extended
img_in layer (and optionally LoRA adapters) are trained while the backbone
is frozen.

Dataset structure:
    dataset/
    ├── sample_000/
    │   ├── image.png          # original clean image (RGB)
    │   ├── mask.png           # binary mask (white=edit, black=keep)
    │   └── prompt.txt         # text caption / editing instruction
    ├── sample_001/
    │   ├── image.png
    │   ├── mask.png
    │   └── prompt.txt
    └── ...

Usage:
    PYTHONPATH=src python scripts/train_inpaint.py \
        --dataset_dir /path/to/dataset \
        --output_dir /path/to/checkpoints \
        --num_epochs 10 \
        --batch_size 1 \
        --lr 1e-5

Requirements:
    pip install torch torchvision einops safetensors pillow fire accelerate
"""

import os
import random
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Setup paths ───────────────────────────────────────────────────────────
WEIGHT_DIR = Path(os.environ.get("FLUX2_WEIGHT_DIR", "/home/diffusion/flux2/weights"))
if (WEIGHT_DIR / "ae.safetensors").exists():
    os.environ["AE_MODEL_PATH"] = str(WEIGHT_DIR / "ae.safetensors")

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    prepare_mask_latent,
)
from flux2.util import load_ae, load_flow_model_inpaint, load_text_encoder


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════


class InpaintDataset(Dataset):
    """
    Loads (image, mask, prompt) triples from a directory structure:
        dataset_dir/sample_XXX/{image.png, mask.png, prompt.txt}
    or a flat structure:
        dataset_dir/{image_XXX.png, mask_XXX.png, prompt_XXX.txt}
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        resolution: int = 512,
        ensure_multiple: int = 16,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.resolution = resolution
        self.ensure_multiple = ensure_multiple

        # Discover samples
        self.samples = self._discover_samples()
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {dataset_dir}")
        print(f"Found {len(self.samples)} training samples in {dataset_dir}")

    def _discover_samples(self) -> list[dict]:
        """Find all (image, mask, prompt) triples."""
        samples = []

        # Try subdirectory structure first
        for subdir in sorted(self.dataset_dir.iterdir()):
            if not subdir.is_dir():
                continue
            image_path = self._find_image(subdir, "image")
            mask_path = self._find_image(subdir, "mask")
            prompt_path = subdir / "prompt.txt"

            if image_path and mask_path and prompt_path.exists():
                samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "prompt": prompt_path,
                })

        # If no subdirs found, try flat structure
        if not samples:
            for img_path in sorted(self.dataset_dir.glob("image_*")):
                stem = img_path.stem.replace("image_", "")
                mask_path = self._find_image(self.dataset_dir, f"mask_{stem}")
                prompt_path = self.dataset_dir / f"prompt_{stem}.txt"
                if mask_path and prompt_path.exists():
                    samples.append({
                        "image": img_path,
                        "mask": mask_path,
                        "prompt": prompt_path,
                    })

        return samples

    @staticmethod
    def _find_image(directory: Path, prefix: str) -> Path | None:
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            p = directory / f"{prefix}{ext}"
            if p.exists():
                return p
        return None

    def _resize_and_crop(self, img: Image.Image) -> Image.Image:
        """Resize to resolution, center crop to multiple of ensure_multiple."""
        w, h = img.size

        # Scale so shortest side = resolution
        scale = self.resolution / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop to multiple of ensure_multiple
        new_w = (new_w // self.ensure_multiple) * self.ensure_multiple
        new_h = (new_h // self.ensure_multiple) * self.ensure_multiple
        left = (img.width - new_w) // 2
        top = (img.height - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image = Image.open(sample["image"]).convert("RGB")
        image = self._resize_and_crop(image)
        image_tensor = T.ToTensor()(image)
        image_tensor = 2.0 * image_tensor - 1.0  # normalize to [-1, 1]

        # Load and preprocess mask
        mask = Image.open(sample["mask"]).convert("L")
        mask = self._resize_and_crop(mask)
        mask = mask.point(lambda p: 255 if p > 128 else 0)  # binarize
        mask_tensor = T.ToTensor()(mask)  # [1, H, W], values in {0, 1}

        # Load prompt
        prompt = sample["prompt"].read_text(encoding="utf-8").strip()

        return {
            "image": image_tensor,    # [3, H, W]
            "mask": mask_tensor,       # [1, H, W]
            "prompt": prompt,          # str
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Flow Matching Loss
# ═══════════════════════════════════════════════════════════════════════════


def flow_matching_loss(
    model: nn.Module,
    ae: nn.Module,
    text_encoder: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the masked Flow Matching loss for inpainting.

    Flow Matching objective:
        loss = || model(x_t, t, cond) - (x_1 - x_0) ||^2
    where x_t = (1 - t) * x_0 + t * x_1
          x_0 = noise, x_1 = clean latent

    For inpainting, loss is ONLY computed in the masked region.
    """
    images = batch["image"].to(device, dtype=torch.bfloat16)   # [B, 3, H, W]
    masks = batch["mask"].to(device, dtype=torch.bfloat16)      # [B, 1, H, W]
    prompts = batch["prompt"]                                    # list[str]

    B = images.shape[0]

    # 1) Encode image to latent space
    with torch.no_grad():
        image_latent = ae.encode(images)  # [B, 128, h, w]

    _, C, latent_h, latent_w = image_latent.shape

    # 2) Downsample mask to latent space
    mask_latent = prepare_mask_latent(masks, latent_h, latent_w)  # [B, 1, h, w]

    # 3) Sample random timestep t ∈ (0, 1)
    t = torch.rand(B, device=device, dtype=torch.bfloat16)
    t = t.clamp(1e-5, 1.0 - 1e-5)  # avoid exact 0 or 1

    # 4) Sample noise x_0
    noise = torch.randn_like(image_latent)

    # 5) Interpolate: x_t = (1-t)*noise + t*clean_latent
    #    (Flow Matching: going from noise to data)
    t_expand = t[:, None, None, None]
    x_t = (1.0 - t_expand) * noise + t_expand * image_latent

    # 6) Target velocity: v = x_1 - x_0 = clean_latent - noise
    target = image_latent - noise

    # 7) Prepare masked image latent (context for inpainting)
    masked_image_latent = image_latent * (1.0 - mask_latent)

    # 8) Flatten everything to token space: [B, C, h, w] → [B, N, C]
    x_t_tokens = rearrange(x_t, "b c h w -> b (h w) c")
    mask_tokens = rearrange(mask_latent, "b c h w -> b (h w) c")
    masked_image_tokens = rearrange(masked_image_latent, "b c h w -> b (h w) c")

    # Build inpaint input: [B, N, 128+1+128 = 257]
    inpaint_input = torch.cat([x_t_tokens, mask_tokens, masked_image_tokens], dim=-1)

    # 9) Create position IDs
    x_dummy = torch.zeros(B, C, latent_h, latent_w, device=device, dtype=torch.bfloat16)
    _, x_ids = batched_prc_img(x_dummy)

    # 10) Encode text
    with torch.no_grad():
        ctx = text_encoder(list(prompts)).to(torch.bfloat16).to(device)
        ctx, ctx_ids = batched_prc_txt(ctx)

    # 11) Timestep + guidance
    guidance_vec = torch.ones(B, device=device, dtype=torch.bfloat16)

    # 12) Forward pass
    pred = model(
        x=inpaint_input,
        x_ids=x_ids,
        timesteps=t,
        ctx=ctx,
        ctx_ids=ctx_ids,
        guidance=guidance_vec,
    )
    # pred: [B, N, 128]

    # 13) Flatten target
    target_tokens = rearrange(target, "b c h w -> b (h w) c")

    # 14) Compute MSE loss, masked to inpaint region only
    loss_per_token = F.mse_loss(pred, target_tokens, reduction="none")  # [B, N, C]

    # mask_tokens: [B, N, 1] → broadcast over C
    masked_loss = loss_per_token * mask_tokens  # zero loss outside mask

    # Normalize: sum over masked tokens / count of masked tokens
    mask_sum = mask_tokens.sum()
    if mask_sum > 0:
        loss = masked_loss.sum() / (mask_sum * C)
    else:
        # Fallback: if mask is empty, use full loss
        loss = loss_per_token.mean()

    return loss


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════


def main(
    dataset_dir: str = "./dataset",
    output_dir: str = "./checkpoints",
    model_name: str = "flux.2-klein-4b",
    # Training hyperparameters
    num_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    # Model options
    train_img_in_only: bool = True,
    use_gradient_checkpointing: bool = True,
    # Data options
    resolution: int = 512,
    # Save options
    save_every_n_steps: int = 500,
    log_every_n_steps: int = 10,
    # Seed
    seed: int = 42,
):
    """
    Fine-tune FLUX.2 for mask-conditioned inpainting.

    Args:
        dataset_dir: Path to dataset directory.
        output_dir: Where to save checkpoints.
        model_name: Which FLUX.2 variant to fine-tune.
        num_epochs: Number of training epochs.
        batch_size: Batch size (1 recommended for large models).
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        gradient_accumulation_steps: Accumulate gradients over N batches.
        max_grad_norm: Max gradient norm for clipping.
        train_img_in_only: If True, only train the img_in layer (fastest).
                           If False, train all parameters (best quality).
        use_gradient_checkpointing: Enable gradient checkpointing (saves VRAM).
        resolution: Training image resolution.
        save_every_n_steps: Save checkpoint every N optimizer steps.
        log_every_n_steps: Print loss every N optimizer steps.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FLUX.2 Inpainting Fine-tuning")
    print("=" * 70)
    print(f"  Model          : {model_name}")
    print(f"  Dataset        : {dataset_dir}")
    print(f"  Resolution     : {resolution}")
    print(f"  Epochs         : {num_epochs}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Grad accum     : {gradient_accumulation_steps}")
    print(f"  Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate  : {lr}")
    print(f"  Train img_in   : {train_img_in_only}")
    print(f"  Output         : {output_dir}")
    print("=" * 70)
    print()

    # ── Load models ───────────────────────────────────────────────────────
    print("[1/4] Loading AutoEncoder (frozen)...")
    ae = load_ae(model_name, device=device)
    ae.eval()
    ae.requires_grad_(False)

    print("[2/4] Loading Text Encoder (frozen)...")
    text_encoder = load_text_encoder(model_name, device=device)
    text_encoder.eval()
    # Freeze text encoder parameters if it's an nn.Module
    if isinstance(text_encoder, nn.Module):
        text_encoder.requires_grad_(False)

    print("[3/4] Loading FLUX.2 Inpaint Model (extended img_in)...")
    model = load_flow_model_inpaint(model_name, debug_mode=False, device=device)

    # Freeze / unfreeze parameters based on training strategy
    if train_img_in_only:
        print("  → Freezing entire backbone, only training img_in layer")
        model.requires_grad_(False)
        model.img_in.requires_grad_(True)
        trainable_params = list(model.img_in.parameters())
    else:
        print("  → Training all model parameters")
        trainable_params = list(model.parameters())

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {num_trainable:,} / {num_total:,} "
          f"({100 * num_trainable / num_total:.2f}%)")

    # Enable gradient checkpointing to save VRAM
    if use_gradient_checkpointing:
        print("  → Enabling gradient checkpointing")
        model.gradient_checkpointing_enable = True  # flag for custom impl

    model.train()

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Loading dataset from {dataset_dir}...")

    def collate_fn(batch_list):
        """Custom collate: stack tensors, keep prompts as list."""
        return {
            "image": torch.stack([b["image"] for b in batch_list]),
            "mask": torch.stack([b["mask"] for b in batch_list]),
            "prompt": [b["prompt"] for b in batch_list],
        }

    dataset = InpaintDataset(dataset_dir, resolution=resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler (cosine)
    total_steps = (len(dataloader) * num_epochs) // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    print(f"\n  Total optimizer steps: {total_steps}")
    print(f"  Batches per epoch   : {len(dataloader)}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Forward + loss
            loss = flow_matching_loss(model, ae, text_encoder, batch, device)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_every_n_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Epoch {epoch+1}/{num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"LR: {current_lr:.2e}"
                    )

                # Save checkpoint
                if global_step % save_every_n_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, out_dir)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n  ✓ Epoch {epoch+1}/{num_epochs} complete | Avg Loss: {avg_epoch_loss:.6f}\n")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, out_dir, tag="best")

        # Save epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, out_dir, tag=f"epoch_{epoch+1}")

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, scheduler, num_epochs, global_step, out_dir, tag="final")
    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  Best loss  : {best_loss:.6f}")
    print(f"  Checkpoints: {out_dir}")
    print("=" * 70)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    out_dir: Path,
    tag: str | None = None,
):
    """Save model checkpoint."""
    from safetensors.torch import save_file

    name = tag if tag else f"step_{global_step}"
    ckpt_dir = out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights as safetensors
    state_dict = model.state_dict()
    save_file(state_dict, str(ckpt_dir / "model.safetensors"))

    # Save optimizer + metadata as torch checkpoint
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, ckpt_dir / "training_state.pt")

    print(f"  💾 Checkpoint saved: {ckpt_dir}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
