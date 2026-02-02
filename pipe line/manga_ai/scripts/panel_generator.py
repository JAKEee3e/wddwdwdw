from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image

from .storyboard import PageSpec, PanelSpec


@dataclass
class SDXLPanelBackend:
    pipe: StableDiffusionXLPipeline


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


def load_sdxl_panel_pipeline(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float16,
) -> SDXLPanelBackend:
    model_dir = Path(model_dir)

    vae = None
    vae_path = model_dir / "vae"
    if vae_path.exists():
        vae = _from_pretrained_with_dtype(AutoencoderKL, vae_path, dtype)

    pipe = _from_pretrained_with_dtype(
        StableDiffusionXLPipeline,
        model_dir,
        dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        vae=vae,
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

    return SDXLPanelBackend(pipe=pipe)


def _panel_size_px(page: PageSpec, panel: PanelSpec) -> tuple[int, int]:
    _, _, w, h = panel.bbox_norm
    width = max(64, int(page.page_width * w))
    height = max(64, int(page.page_height * h))

    width = (width // 8) * 8
    height = (height // 8) * 8
    return max(64, width), max(64, height)


@torch.inference_mode()
def generate_page_panels(
    backend: SDXLPanelBackend,
    page: PageSpec,
    out_dir: str | Path,
    global_negative_prompt: str,
    default_seed: int = 1234,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}

    for panel in page.panels:
        w, h = _panel_size_px(page, panel)
        seed = panel.seed if panel.seed is not None else default_seed
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        prompt = panel.sdxl_prompt
        neg = " , ".join([panel.sdxl_negative_prompt, global_negative_prompt]).strip(" ,")

        image = backend.pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=w,
            height=h,
            num_inference_steps=int(panel.steps),
            guidance_scale=float(panel.cfg_scale),
            generator=gen,
        ).images[0]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        path = out_dir / f"{panel.panel_id}.png"
        image.save(path)
        results[panel.panel_id] = path

    return results
