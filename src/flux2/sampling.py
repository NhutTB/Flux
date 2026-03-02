import math

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from PIL import Image
from torch import Tensor

from .model import Flux2


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def encode_image_refs(ae, img_ctx: list[Image.Image]):
    scale = 10

    if len(img_ctx) > 1:
        limit_pixels = 1024**2
    elif len(img_ctx) == 1:
        limit_pixels = 2024**2
    else:
        limit_pixels = None

    if not img_ctx:
        return None, None

    img_ctx_prep = default_prep(img=img_ctx, limit_pixels=limit_pixels)
    if not isinstance(img_ctx_prep, list):
        img_ctx_prep = [img_ctx_prep]

    # Encode each reference image
    encoded_refs = []
    for img in img_ctx_prep:
        encoded = ae.encode(img[None].cuda())[0]
        encoded_refs.append(encoded)

    # Create time offsets for each reference
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    t_off = [t.view(-1) for t in t_off]

    # Process with position IDs
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)

    # Concatenate all references along sequence dimension
    ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
    ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0)  # (1, total_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

    return ref_tokens.to(torch.bfloat16), ref_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape  # noqa: F841

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy dimension
        "w": torch.arange(1),  # dummy dimension
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)

    return batched_prc


def listed_wrapper(fn):
    def listed_prc(
        x: list[Tensor],
        t_coord: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return list(x), list(x_ids)

    return listed_prc


def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape  # noqa: F841
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


listed_prc_img = listed_wrapper(prc_img)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def center_crop_to_multiple_of_x(
    img: Image.Image | list[Image.Image], x: int
) -> Image.Image | list[Image.Image]:
    if isinstance(img, list):
        return [center_crop_to_multiple_of_x(_img, x) for _img in img]  # type: ignore

    w, h = img.size
    new_w = (w // x) * x
    new_h = (h // x) * x

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    resized = img.crop((left, top, right, bottom))
    return resized


def cap_pixels(img: Image.Image | list[Image.Image], k):
    if isinstance(img, list):
        return [cap_pixels(_img, k) for _img in img]
    w, h = img.size
    pixel_count = w * h

    if pixel_count <= k:
        return img

    # Scaling factor to reduce total pixels below K
    scale = math.sqrt(k / pixel_count)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(img: Image.Image | list[Image.Image], max_ar=8, min_sidelength=64):
    if isinstance(img, list):
        return [cap_min_pixels(_img, max_ar=max_ar, min_sidelength=min_sidelength) for _img in img]
    w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(f"Skipping due to minimal sidelength underschritten h {h} w {w}")
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Skipping due to maximal ar overschritten h {h} w {w}")
    return img


def to_rgb(img: Image.Image | list[Image.Image]):
    if isinstance(img, list):
        return [
            to_rgb(
                _img,
            )
            for _img in img
        ]
    return img.convert("RGB")


def default_images_prep(
    x: Image.Image | list[Image.Image],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(x, list):
        return [default_images_prep(e) for e in x]  # type: ignore
    x_tensor = torchvision.transforms.ToTensor()(x)
    return 2 * x_tensor - 1


def default_prep(
    img: Image.Image | list[Image.Image], limit_pixels: int | None, ensure_multiple: int = 16
) -> torch.Tensor | list[torch.Tensor]:
    img_rgb = to_rgb(img)
    img_min = cap_min_pixels(img_rgb)  # type: ignore
    if limit_pixels is not None:
        img_cap = cap_pixels(img_min, limit_pixels)  # type: ignore
    else:
        img_cap = img_min
    img_crop = center_crop_to_multiple_of_x(img_cap, ensure_multiple)  # type: ignore
    img_tensor = default_images_prep(img_crop)
    return img_tensor


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def denoise(
    model: Flux2,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


def denoise_with_mask_blending(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    mask_tokens: Tensor,
    image_latent_tokens: Tensor,
    noise_tokens: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    """
    Denoising loop with latent-space mask blending (RePaint/SDEdit style).
    Works with the STANDARD model — no fine-tuning needed.

    At each step:
      1. Replace unmasked region with correctly-noised original
      2. Model predicts velocity for the full image
      3. Euler step update

    Args:
        mask_tokens:         [B, N, 1]  flattened mask (1=edit, 0=keep)
        image_latent_tokens: [B, N, C]  flattened clean image latent
        noise_tokens:        [B, N, C]  the SAME initial noise used to create img
    """
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        # Replace unmasked region with noised original at current timestep
        # FLUX convention: schedule 1.0→0.0, t=1 is noise, t=0 is clean
        # So: x_t = t * noise + (1-t) * clean
        noised_original = t_curr * noise_tokens + (1.0 - t_curr) * image_latent_tokens
        img = mask_tokens * img + (1.0 - mask_tokens) * noised_original

        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )

        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    # Final: ensure unmasked region is exactly clean original
    img = mask_tokens * img + (1.0 - mask_tokens) * image_latent_tokens
    return img


def vanilla_guidance(x: torch.Tensor, cfg_val: float) -> torch.Tensor:
    x_u, x_c = x.chunk(2)
    x = x_u + cfg_val * (x_c - x_u)
    return x


def denoise_cfg(
    model: Flux2,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,  # Already cat([txt_empty, txt_prompt])
    txt_ids: Tensor,
    timesteps: list[float],
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    img = torch.cat([img, img], dim=0)
    img_ids = torch.cat([img_ids, img_ids], dim=0)

    if img_cond_seq is not None:
        assert img_cond_seq_ids is not None
        img_cond_seq = torch.cat([img_cond_seq, img_cond_seq], dim=0)
        img_cond_seq_ids = torch.cat([img_cond_seq_ids, img_cond_seq_ids], dim=0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=None,
        )

        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]

        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = torch.cat([pred, pred], dim=0)

        img = img + (t_prev - t_curr) * pred

    return img.chunk(2)[0]


# ── Inpainting helpers ────────────────────────────────────────────────────


def prepare_mask_latent(
    mask: Tensor,
    latent_h: int,
    latent_w: int,
) -> Tensor:
    """
    Downsample a binary mask to latent spatial dimensions.

    Args:
        mask: [B, 1, H, W] binary mask in pixel space (1=edit, 0=keep).
        latent_h: Height in latent space.
        latent_w: Width  in latent space.

    Returns:
        mask_latent: [B, 1, latent_h, latent_w] float mask in latent space.
    """
    original_dtype = mask.dtype
    mask_latent = F.interpolate(
        mask.float(),
        size=(latent_h, latent_w),
        mode="nearest",
    )
    return mask_latent.to(original_dtype)


def prepare_inpaint_latent(
    noisy_latent: Tensor,
    mask_latent: Tensor,
    image_latent: Tensor,
) -> Tensor:
    """
    Concatenate noisy latent + mask + masked_image_latent along the channel
    dimension, producing the full inpaint input.

    Args:
        noisy_latent:  [B, C, H, W]  – the current noisy sample  (C=128)
        mask_latent:   [B, 1, H, W]  – binary mask in latent space
        image_latent:  [B, C, H, W]  – encoded clean image latent (C=128)

    Returns:
        concat:  [B, 2*C+1, H, W]  (e.g. 257 channels)
    """
    # Zero out inpaint region so model sees clean context outside mask
    masked_image_latent = image_latent * (1.0 - mask_latent)
    return torch.cat([noisy_latent, mask_latent, masked_image_latent], dim=1)


def denoise_inpaint(
    model: Flux2,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    # inpaint conditioning (all in *flattened-token* space already prc_img'd)
    mask_tokens: Tensor,
    image_latent_tokens: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float,
    # extra img tokens (reference images)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    """
    Denoising loop for channel-concatenation inpainting.

    At every step the model receives  [noisy_tokens | mask_tokens | masked_image_tokens]
    concatenated along the *channel* (last) dimension of the token sequence.

    Args:
        img:                 [B, N, C]     – noisy latent tokens (N = h*w, C = 128)
        img_ids:             [B, N, 4]     – positional ids
        txt / txt_ids:       text embeddings & ids
        mask_tokens:         [B, N, 1]     – flattened mask   (same N)
        image_latent_tokens: [B, N, C]     – flattened clean image latent
        timesteps:           schedule of t values
        guidance:            guidance scale
        img_cond_seq / ids:  optional reference image tokens
    """
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    # Pre-compute masked image tokens (zero-out edit region)
    masked_image_tokens = image_latent_tokens * (1.0 - mask_tokens)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        # Build inpaint input: concat along channel dim
        # img: [B, N, 128], mask_tokens: [B, N, 1], masked: [B, N, 128]
        # → inpaint_input: [B, N, 257]
        inpaint_input = torch.cat(
            [img, mask_tokens, masked_image_tokens], dim=-1
        )

        img_input = inpaint_input
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            # Pad reference tokens to match inpaint channel width (257)
            pad_width = inpaint_input.shape[-1] - img_cond_seq.shape[-1]
            if pad_width > 0:
                ref_pad = torch.zeros(
                    img_cond_seq.shape[0],
                    img_cond_seq.shape[1],
                    pad_width,
                    device=img_cond_seq.device,
                    dtype=img_cond_seq.dtype,
                )
                img_cond_seq_padded = torch.cat(
                    [img_cond_seq, ref_pad], dim=-1
                )
            else:
                img_cond_seq_padded = img_cond_seq
            img_input = torch.cat((img_input, img_cond_seq_padded), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )

        # Only keep prediction for the actual image tokens
        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


# ── RF-Inversion ──────────────────────────────────────────────────────────


def invert(
    model: Flux2,
    img: Tensor,          # [B, N, C] clean image latent tokens (x_0)
    img_ids: Tensor,      # [B, N, 4]
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: list[float],   # get_schedule() output: high→low (denoising order)
    guidance: float,
    gamma: float = 0.5,       # 0=pure model velocity, 1=straight line to noise
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
) -> list[Tensor]:
    """
    RF-Inversion: forward ODE từ x_0 (clean) → x_T (noise).

    FLUX flow matching: x_t = (1-t)*x_0 + t*eps, nên forward ODE là:
        dx/dt = v_θ(x_t, t)  với dt > 0 (t tăng dần từ 0→1)

    gamma blending: cân bằng model velocity với straight-line path đến noise.
        v_eff = (1-gamma)*v_model + gamma*(eps - x_0)
        gamma=0 → pure model (edit tự do, ít bám structure)
        gamma=1 → straight path (bám sát ảnh gốc)

    Args:
        img:       [B, N, C]  clean image latent tokens (x_0), output của ae.encode
        timesteps: từ get_schedule() — denoising order (t_high→0).
                   Inversion chạy NGƯỢC: 0→t_high

    Returns:
        trajectory: list[Tensor] độ dài len(timesteps).
                    trajectory[0] = x_0 (clean)
                    trajectory[-1] = x_T (inverted noise)
                    trajectory[i] ↔ inv_timesteps[i] = timesteps[-(i+1)]
    """
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # Sample noise target (x_T đích đến của inversion)
    eps = torch.randn_like(img)

    # Inversion chạy ngược schedule: 0.0 → t_max
    inv_timesteps = list(reversed(timesteps))  # [0.0, ..., t_high]

    trajectory = [img.clone()]  # trajectory[0] = x_0
    x = img.clone()

    for t_curr, t_next in zip(inv_timesteps[:-1], inv_timesteps[1:]):
        t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
        dt = t_next - t_curr  # > 0

        img_input = x
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_cond_seq is not None:
            pred = pred[:, : x.shape[1]]

        # Straight-line velocity: hướng từ x_0 đến eps
        v_straight = eps - trajectory[0]

        # Gamma blend
        v_eff = (1.0 - gamma) * pred + gamma * v_straight

        x = x + dt * v_eff
        trajectory.append(x.clone())

    return trajectory  # len = len(timesteps)


def denoise_rf_inversion_inpaint(
    model: Flux2,
    img: Tensor,                     # [B, N, C] x_T (= trajectory[-1])
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    mask_tokens: Tensor,             # [B, N, 1]  1=edit, 0=keep
    trajectory: list[Tensor],        # từ invert(), [0]=x_0 clean, [-1]=x_T
    timesteps: list[float],          # denoising order high→low
    guidance: float,
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
) -> Tensor:
    """
    Denoising với RF-Inversion mask blending.

    Tại mỗi bước i (t đi từ t_high → 0):
      unmasked region = trajectory[N-1-i]   (exact x_t của ảnh gốc, không approximate)
      masked region   = current denoised x  (model generates new content)

    Khác RePaint: trajectory[t] là CHÍNH XÁC x_t của ảnh gốc dưới forward ODE,
    không phải t*noise + (1-t)*clean (approximation có thể sai với FLUX schedule).
    """
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    n = len(trajectory)  # = len(timesteps)

    x = img.clone()  # bắt đầu từ x_T

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # Map step → trajectory index
        # i=0 (t_curr=t_high) → trajectory[-1] = x_T
        # i=N-2 (t_curr≈0+)  → trajectory[1]
        traj_idx = max(0, min(n - 1 - i, n - 1))
        x_traj = trajectory[traj_idx].to(x.device, x.dtype)

        # Blend TRƯỚC khi predict: unmasked = trajectory, masked = current x
        x = mask_tokens * x + (1.0 - mask_tokens) * x_traj

        t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

        img_input = x
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_cond_seq is not None:
            pred = pred[:, : x.shape[1]]

        x = x + (t_prev - t_curr) * pred

    # Final: enforce unmasked = exact x_0 clean
    x0 = trajectory[0].to(x.device, x.dtype)
    x = mask_tokens * x + (1.0 - mask_tokens) * x0
    return x


def concatenate_images(
    images: list[Image.Image],
) -> Image.Image:
    """
    Concatenate a list of PIL images horizontally with center alignment and white background.
    """

    # If only one image, return a copy of it
    if len(images) == 1:
        return images[0].copy()

    # Convert all images to RGB if not already
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    # Calculate dimensions for horizontal concatenation
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new image with white background
    background_color = (255, 255, 255)
    new_img = Image.new("RGB", (total_width, max_height), background_color)

    # Paste images with center alignment
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return new_img