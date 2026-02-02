from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from .pipeline import Backends, default_model_paths, init_backends, run_one_page


class MangaApp:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._backends: Optional[Backends] = None

    def init_models(self) -> None:
        with self._lock:
            if self._backends is not None:
                return
            paths = default_model_paths(self.root)
            self._backends = init_backends(paths)

    def generate(self, prompt: str, pages: int) -> Tuple[Image.Image, str]:
        if not prompt or not prompt.strip():
            raise gr.Error("Prompt is empty")

        self.init_models()

        with self._lock:
            out_path = run_one_page(
                root=self.root,
                story_prompt=prompt,
                pages=int(pages),
                page_index=0,
                backends=self._backends,
            )

        img = Image.open(out_path).convert("RGB")
        return img, str(out_path)


def build_gradio(root: str | Path) -> gr.Blocks:
    app = MangaApp(root)

    with gr.Blocks(title="Local Manga AI") as demo:
        gr.Markdown("# Local Manga AI (Qwen2.5 → SDXL → SAM → SDXL Inpaint → Composer)")

        with gr.Row():
            prompt = gr.Textbox(
                label="Story Prompt",
                lines=6,
                placeholder="A short story prompt for one manga page...",
            )

        with gr.Row():
            pages = gr.Slider(label="Pages (currently generates page 1)", minimum=1, maximum=4, step=1, value=1)

        run_btn = gr.Button("Generate")

        with gr.Row():
            preview = gr.Image(label="Page Preview", type="pil")

        out_file = gr.File(label="Download")

        run_btn.click(fn=app.generate, inputs=[prompt, pages], outputs=[preview, out_file])

    return demo


def launch_gradio(root: str | Path, host: str = "0.0.0.0", port: int = 7860):
    demo = build_gradio(root)
    return demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
    )
