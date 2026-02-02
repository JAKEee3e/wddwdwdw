from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SDXLInpaintBackend:
    pipe: StableDiffusionXLInpaintPipeline


def _from_pretrained_with_dtype(cls, model_dir: Path, dtype: torch.dtype, **kwargs):
    try:
        sig = inspect.signature(cls.from_pretrained)
        if "dtype" in sig.parameters:
            kwargs["dtype"] = dtype
        elif "torch_dtype" in sig.parameters:
            kwargs["torch_dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    except Exception:
        kwargs["torch_dtype"] = dtype

    return cls.from_pretrained(model_dir, **kwargs)


def load_sdxl_inpaint_pipeline(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float16,
) -> SDXLInpaintBackend:
    model_dir = Path(model_dir)

    pipe = _from_pretrained_with_dtype(
        StableDiffusionXLInpaintPipeline,
        model_dir,
        dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if torch.cuda.is_available():
        pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        else:
            pipe.enable_vae_tiling()
    except Exception:
        pass

    return SDXLInpaintBackend(pipe=pipe)


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    words = text.split()
    if not words:
        return text

    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int) -> tuple[ImageFont.ImageFont, str]:
    base_sizes = [44, 40, 36, 32, 28, 24, 22, 20, 18, 16]

    for size in base_sizes:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            font = ImageFont.load_default()

        wrapped = _wrap_text_to_width(draw, text=text, font=font, max_w=int(box_w * 0.92))
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=int(size * 0.25), align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= box_w * 0.95 and th <= box_h * 0.85:
            return font, wrapped

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    wrapped = _wrap_text_to_width(draw, text=text, font=font, max_w=int(box_w * 0.92))
    return font, wrapped


def _draw_text_in_ellipse(img: Image.Image, mask: Image.Image, text: str) -> Image.Image:
    rgba = img.convert("RGBA")
    m = mask.convert("L")
    arr = np.array(m)
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return rgba

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad = max(6, int(min(x1 - x0, y1 - y0) * 0.08))
    x0 = min(max(0, x0 + pad), rgba.width - 1)
    y0 = min(max(0, y0 + pad), rgba.height - 1)
    x1 = min(max(0, x1 - pad), rgba.width - 1)
    y1 = min(max(0, y1 - pad), rgba.height - 1)

    box_w = max(32, x1 - x0)
    box_h = max(32, y1 - y0)

    draw = ImageDraw.Draw(rgba)
    font, wrapped = _fit_font(draw, text=text, box_w=box_w, box_h=box_h)

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = x0 + (box_w - tw) // 2
    ty = y0 + (box_h - th) // 2

    draw.multiline_text(
        (tx, ty),
        wrapped,
        font=font,
        fill=(0, 0, 0, 255),
        spacing=6,
        align="center",
        stroke_width=2,
        stroke_fill=(255, 255, 255, 255),
    )

    return rgba


@torch.inference_mode()
def inpaint_bubbles(
    backend: SDXLInpaintBackend,
    panel_image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int = 28,
    cfg_scale: float = 5.5,
    seed: int = 1234,
) -> Image.Image:
    if panel_image.mode != "RGB":
        panel_image = panel_image.convert("RGB")

    m = mask.convert("L")

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    out = backend.pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=panel_image,
        mask_image=m,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg_scale),
        generator=gen,
    ).images[0]

    if not isinstance(out, Image.Image):
        out = Image.fromarray(out)

    return out


def inpaint_bubbles_and_add_text(
    backend: SDXLInpaintBackend,
    panel_image: Image.Image,
    mask: Image.Image,
    text: str,
    seed: int,
) -> Image.Image:
    bubble_prompt = (
        "clean white manga speech bubble, crisp black outline, high contrast, "
        "flat white fill, professional comic inking, no text"
    )
    bubble_neg = "text, letters, watermark, logo, signature, blurry, lowres"

    bubble_img = inpaint_bubbles(
        backend=backend,
        panel_image=panel_image,
        mask=mask,
        prompt=bubble_prompt,
        negative_prompt=bubble_neg,
        steps=26,
        cfg_scale=5.0,
        seed=seed,
    )

    return _draw_text_in_ellipse(bubble_img, mask=mask, text=text).convert("RGB")
