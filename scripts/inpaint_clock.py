r"""
Inpainting script for FLUX.2-klein-4B with mask-conditioned generation.

This script supports two modes:
  1. **Reference-based editing** (no mask): Uses the input image as reference
     context and generates a new image guided by the prompt.
  2. **Mask-conditioned inpainting** (with mask): Uses the Channel Concatenation
     approach — the DiT backbone receives [noisy_latent | mask | masked_image_latent]
     so it is explicitly aware of which region to edit.

Usage (without mask — reference-based editing):
    PYTHONPATH=src python scripts/inpaint_clock.py \
        --input_image test.jpg \
        --prompt "Add a wall clock on the wall"

Usage (with mask — targeted inpainting via channel concat):
    PYTHONPATH=src python scripts/inpaint_clock.py \
        --input_image test.jpg \
        --mask_image mask.png \
        --prompt "A vintage round wall clock"

Mask format:
    - WHITE (255) = area to EDIT (inpaint new content)
    - BLACK (0)   = area to KEEP (preserve original)

Weights are auto-downloaded from HuggingFace.
"""

import os
import random
import sys
from pathlib import Path

import torch
import torchvision
from einops import rearrange
from PIL import ExifTags, Image
from torch import Tensor

# ── Weights will be auto-downloaded from HuggingFace ──────────────────────
WEIGHT_DIR = Path("/home/diffusion/flux2/weights")
os.environ["AE_MODEL_PATH"] = str(WEIGHT_DIR / "ae.safetensors")
from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_with_mask_blending,
    denoise_rf_inversion_inpaint,
    encode_image_refs,
    get_schedule,
    invert,
    prepare_mask_latent,
    scatter_ids,
)
from flux2.util import load_ae, load_flow_model, load_text_encoder
from flux2.model import Flux2


# ── Pixel-space mask compositing ──────────────────────────────────────────


def prepare_pixel_mask(
    mask_path: str | Path,
    width: int,
    height: int,
    feather_radius: int = 12,
) -> Image.Image:
    """
    Load mask and apply Gaussian feathering for smooth edges.

    Args:
        mask_path: Path to mask (white=edit, black=keep).
        width, height: Target dimensions.
        feather_radius: Gaussian blur radius for soft edges.

    Returns:
        PIL Image in "L" mode with feathered mask.
    """
    from PIL import ImageFilter

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.Resampling.NEAREST)

    # Binarize
    mask = mask.point(lambda p: 255 if p > 128 else 0)

    # Apply Gaussian blur for feathered edges
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return mask


def composite_with_mask(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Composite generated content onto original using mask.
    White mask = use generated, Black mask = use original.
    """
    return Image.composite(generated, original, mask)


# ── Main ──────────────────────────────────────────────────────────────────


def main(
    input_image: str = "test.jpg",
    mask_image: str | None = None,
    prompt: str = "Add a standing lamp on the right side of the image",
    output_dir: str = "output",
    seed: int | None = None,
    num_steps: int = 4,
    guidance: float = 1.0,
    cpu_offloading: bool = False,
    inpaint_strength: float = 1.0,
    use_inpaint_model: bool = False,
    upsample: bool = False,
    use_rf_inversion: bool = True,
    gamma: float = 0.5,
):
    """
    Run FLUX.2-klein-4B image editing with optional mask-based inpainting.

    Args:
        input_image: Path to the input image.
        mask_image: Path to mask image (white=edit, black=keep). If None,
                    runs in reference-based editing mode (no mask).
        prompt: The editing prompt.
        output_dir: Directory to save the output image.
        seed: Random seed. None = random.
        num_steps: Number of denoising steps (4 for distilled klein).
        guidance: Guidance scale (1.0 for distilled klein).
        cpu_offloading: Offload models to CPU when not in use (saves VRAM).
        inpaint_strength: Chỉ dùng cho mask_blend mode (0.0-1.0).
        use_inpaint_model: Dùng model 257ch channel-concat (cần fine-tune).
        use_rf_inversion: Dùng RF-Inversion thay vì RePaint (recommended).
        gamma: RF-Inversion identity preservation.
               0.0=creative (edit mạnh), 1.0=faithful (giữ sát ảnh gốc).
               Bắt đầu test với 0.5.
    """

    model_name = "flux.2-klein-4b"
    torch_device = torch.device("cuda")

    # ── Validate inputs ───────────────────────────────────────────────────
    input_path = Path(input_image)
    if not input_path.exists():
        print(f"ERROR: Input image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    use_mask = mask_image is not None
    if use_mask:
        mask_path = Path(mask_image)
        if not mask_path.exists():
            print(f"ERROR: Mask image not found: {mask_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Input image : {input_path}")
    print(f"Mask image  : {mask_path if use_mask else 'None (reference-based editing)'}")
    print(f"Prompt      : {prompt}")
    print(f"Model       : {model_name}")
    print(f"Steps       : {num_steps}")
    print(f"Guidance    : {guidance}")
    if use_mask:
        print(f"Strength    : {inpaint_strength}")
        mode_str = f"RF-Inversion (gamma={gamma})" if use_rf_inversion else "mask_blend (RePaint)"
        print(f"Inpaint mode: {mode_str}")
    print()

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading text encoder...")
    text_encoder = load_text_encoder(model_name, device=torch_device)

    print("Loading flow model (FLUX.2-klein-4B)...")
    model = load_flow_model(
        model_name,
        debug_mode=False,
        device="cpu" if cpu_offloading else torch_device,
    )

    print("Loading autoencoder...")
    ae = load_ae(model_name)
    ae.eval()
    text_encoder.eval()

    # ── Prepare output directory ──────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    output_name = out_dir / f"inpainted_{len(list(out_dir.glob('*')))}.png"

    # ── Set seed ──────────────────────────────────────────────────────────
    if seed is None:
        seed = random.randrange(2**31)
    print(f"Seed        : {seed}")
    print()

    # ── Open and prepare input image ──────────────────────────────────────
    img_ref = Image.open(input_path).convert("RGB")
    width, height = img_ref.size
    print(f"Image size  : {width} x {height}")

    # Ensure dimensions are multiples of 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    if (width, height) != img_ref.size:
        img_ref = img_ref.resize((width, height), Image.Resampling.LANCZOS)
        print(f"Resized to  : {width} x {height} (multiple of 16)")
    print()

    latent_h = height // 16
    latent_w = width // 16

    # ── Run inference ─────────────────────────────────────────────────────
    print("Starting inference...")
    with torch.no_grad():
        # 1) Encode reference image
        print("  Encoding reference image...")
        ref_tokens, ref_ids = encode_image_refs(ae, [img_ref])

        # 2) Upsample prompt (optional) + encode text
        if upsample:
            print("  Upsampling prompt (enriching with Qwen3)...")
            upsampled = text_encoder.upsample_prompt(
                [prompt], is_editing=use_mask,
            )
            prompt = upsampled[0]
            print(f"  Upsampled : {prompt}")

        print("  Encoding text prompt...")
        ctx = text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offloading:
            text_encoder = text_encoder.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # 3) Create initial noise
        print("  Creating noise tensor...")
        shape = (1, 128, latent_h, latent_w)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")

        # 4) Get denoising schedule
        x_tokens, x_ids = batched_prc_img(x)
        timesteps = get_schedule(num_steps, x_tokens.shape[1])

        # ── INPAINTING PATH ───────────────────────────────────────────────
        if use_mask:
            # Encode ảnh gốc vào latent space
            img_ref_tensor = torchvision.transforms.ToTensor()(img_ref).unsqueeze(0)
            img_ref_tensor = (2 * img_ref_tensor - 1).cuda().float()
            image_latent = ae.encode(img_ref_tensor).to(torch.bfloat16)

            # Prepare mask
            mask_pil = Image.open(mask_path).convert("L")
            mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
            mask_pil = mask_pil.point(lambda p: 255 if p > 128 else 0)
            mask_tensor = torchvision.transforms.ToTensor()(mask_pil).unsqueeze(0)
            mask_tensor = mask_tensor.to(torch.bfloat16).cuda()
            mask_latent = prepare_mask_latent(mask_tensor, latent_h, latent_w)

            # Debug mask coverage
            mask_nonzero = (mask_latent > 0.5).sum().item()
            total_tokens = latent_h * latent_w
            print(f"  [Mask] Latent {latent_h}x{latent_w} — masked: "
                  f"{mask_nonzero}/{total_tokens} ({100*mask_nonzero/total_tokens:.1f}%)")
            if mask_nonzero == 0:
                print("  [WARNING] Mask EMPTY! Kiểm tra mask.png.")

            mask_tokens = rearrange(mask_latent, "b c h w -> b (h w) c")
            image_latent_tokens = rearrange(image_latent, "b c h w -> b (h w) c")

            # ── RF-Inversion mode ──────────────────────────────────────────
            if use_rf_inversion:
                print(f"  Inverting original image (forward ODE, gamma={gamma})...")
                img_latent_tokens, img_latent_ids = batched_prc_img(image_latent)

                trajectory = invert(
                    model,
                    img_latent_tokens,
                    img_latent_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    gamma=gamma,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
                print(f"  Denoising with RF-Inversion ({num_steps} steps)...")
                x_tokens = denoise_rf_inversion_inpaint(
                    model,
                    trajectory[-1],    # x_T từ inversion
                    img_latent_ids,
                    ctx,
                    ctx_ids,
                    mask_tokens=mask_tokens,
                    trajectory=trajectory,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
                # Dùng img_latent_ids (không phải x_ids từ noise) để scatter
                x_ids = img_latent_ids

            # ── Mask blend (RePaint) mode ──────────────────────────────────
            else:
                noise_tokens = rearrange(x, "b c h w -> b (h w) c")
                print(f"  Denoising mask blending / RePaint ({num_steps} steps)...")
                x_tokens = denoise_with_mask_blending(
                    model,
                    x_tokens,
                    x_ids,
                    ctx,
                    ctx_ids,
                    mask_tokens=mask_tokens,
                    image_latent_tokens=image_latent_tokens,
                    noise_tokens=noise_tokens,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )

        # ── STANDARD PATH (reference-based, no mask) ──────────────────────
        else:
            print(f"  Denoising ({num_steps} steps)...")
            x_tokens = denoise(
                model,
                x_tokens,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )

        # 7) Scatter IDs and decode
        print("  Decoding latent...")
        x_decoded = torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)
        x_decoded = ae.decode(x_decoded).float()

        if cpu_offloading:
            model = model.cpu()
            torch.cuda.empty_cache()

    # ── Post-process ──────────────────────────────────────────────────────
    x_decoded = x_decoded.clamp(-1, 1)
    x_decoded = rearrange(x_decoded[0], "c h w -> h w c")
    img_generated = Image.fromarray((127.5 * (x_decoded + 1.0)).cpu().byte().numpy())

    # ── Apply mask compositing in pixel space ─────────────────────────────
    if use_mask:
        print("  Applying pixel-space mask compositing (feathered edges)...")
        pixel_mask = prepare_pixel_mask(mask_path, width, height, feather_radius=12)
        # White mask = use generated, Black mask = use original
        img_out = composite_with_mask(img_ref, img_generated, pixel_mask)
    else:
        img_out = img_generated

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;flux2"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    img_out.save(output_name, exif=exif_data, quality=95, subsampling=0)

    print()
    print(f"Done! Output saved to: {output_name}")
    print()

    # ── Show side-by-side comparison ──────────────────────────────────────
    from flux2.sampling import concatenate_images

    images_to_compare = [img_ref.resize((width, height))]
    if use_mask:
        # Show mask visualization too
        mask_viz = Image.open(mask_path).convert("RGB").resize((width, height))
        images_to_compare.append(mask_viz)
    images_to_compare.append(img_out)

    comparison = concatenate_images(images_to_compare)
    comparison_path = out_dir / f"comparison_{output_name.stem}.png"
    comparison.save(comparison_path, quality=95)
    print(f"Side-by-side comparison saved to: {comparison_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)