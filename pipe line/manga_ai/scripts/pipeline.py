from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

from PIL import Image

from .model_downloader import ensure_models_downloaded

if TYPE_CHECKING:
    from . import bubble_detector, bubble_inpainter, panel_generator, storyboard


@dataclass
class ModelPaths:
    qwen_dir: Path
    sdxl_dir: Path
    sdxl_inpaint_dir: Path
    sam_ckpt: Path


@dataclass
class Backends:
    qwen: "storyboard.QwenBackend"
    sdxl_panel: "panel_generator.SDXLPanelBackend"
    sam: "bubble_detector.SAMBubbleBackend"
    sdxl_inpaint: "bubble_inpainter.SDXLInpaintBackend"


def init_backends(paths: ModelPaths) -> Backends:
    ensure_models_downloaded(
        qwen_dir=paths.qwen_dir,
        sdxl_dir=paths.sdxl_dir,
        sdxl_inpaint_dir=paths.sdxl_inpaint_dir,
        sam_ckpt=paths.sam_ckpt,
    )

    from . import bubble_detector, bubble_inpainter, panel_generator, storyboard

    qwen = storyboard.load_qwen_backend(paths.qwen_dir)
    sdxl_panel = panel_generator.load_sdxl_panel_pipeline(paths.sdxl_dir)
    sam = bubble_detector.load_sam(paths.sam_ckpt)
    sdxl_inpaint = bubble_inpainter.load_sdxl_inpaint_pipeline(paths.sdxl_inpaint_dir)
    return Backends(qwen=qwen, sdxl_panel=sdxl_panel, sam=sam, sdxl_inpaint=sdxl_inpaint)


def default_model_paths(root: str | Path) -> ModelPaths:
    root = Path(root)

    qwen_dir = os.environ.get("MANGA_AI_QWEN_DIR")
    sdxl_dir = os.environ.get("MANGA_AI_SDXL_DIR")
    sdxl_inpaint_dir = os.environ.get("MANGA_AI_SDXL_INPAINT_DIR")
    sam_ckpt = os.environ.get("MANGA_AI_SAM_CKPT")

    return ModelPaths(
        qwen_dir=Path(qwen_dir) if qwen_dir else (root / "models" / "qwen2.5"),
        sdxl_dir=Path(sdxl_dir) if sdxl_dir else (root / "models" / "sdxl"),
        sdxl_inpaint_dir=Path(sdxl_inpaint_dir) if sdxl_inpaint_dir else (root / "models" / "sdxl_inpaint"),
        sam_ckpt=Path(sam_ckpt) if sam_ckpt else (root / "models" / "sam" / "sam_vit_h_4b8939.pth"),
    )


def run_one_page(
    root: str | Path,
    story_prompt: str,
    page_index: int = 0,
    pages: int = 1,
    backends: Optional[Backends] = None,
) -> Path:
    from . import bubble_detector, bubble_inpainter, page_composer, panel_generator, storyboard

    root = Path(root)
    paths = default_model_paths(root)

    if backends is None:
        backends = init_backends(paths)

    sb = storyboard.generate_storyboard_from_backend(
        story_prompt=story_prompt,
        backend=backends.qwen,
        pages=pages,
    )

    page = sb.pages[page_index]

    storyboard.save_storyboard(sb, root / "outputs" / "storyboards" / "storyboard.json")

    panel_dir = root / "outputs" / "panels" / f"page_{page.page_index:03d}"
    panel_paths = panel_generator.generate_page_panels(
        backend=backends.sdxl_panel,
        page=page,
        out_dir=panel_dir,
        global_negative_prompt=sb.global_negative_prompt,
    )

    final_panels: Dict[str, Image.Image] = {}

    for panel in page.panels:
        img = Image.open(panel_paths[panel.panel_id]).convert("RGB")

        if panel.dialogue:
            bboxes = bubble_detector.suggest_bubble_bboxes(
                panel_image=img,
                dialogue=panel.dialogue,
                backend=backends.sam,
            )
            masks = bubble_detector.bubble_masks_from_bboxes(img.size, bboxes=bboxes)

            for i, d in enumerate(panel.dialogue):
                if i >= len(masks):
                    break
                img = bubble_inpainter.inpaint_bubbles_and_add_text(
                    backend=backends.sdxl_inpaint,
                    panel_image=img,
                    mask=masks[i],
                    text=d.text,
                    seed=(panel.seed or 1234) + 999 + i,
                )

        final_panels[panel.panel_id] = img

    out_page = root / "outputs" / "pages" / f"page_{page.page_index:03d}.png"
    return page_composer.compose_page(
        page=page,
        panel_images=final_panels,
        out_path=out_page,
    )
