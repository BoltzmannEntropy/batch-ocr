import gradio as gr
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import tempfile
import traceback
import numpy as np

from paddleocr import PaddleOCR, PPStructureV3
import paddle


# ==============================
# Globals
# ==============================
ocr_instance = None
doc_pipeline = None


def initialize_pipelines(
    use_gpu: bool = True,
    lang: str = "en",
    use_orientation: bool = True,
    use_unwarp: bool = True,
    use_textline: bool = True,
    use_chart: bool = False,
):
    """Initialize PaddleOCR (text) and PP-StructureV3 (document parsing)."""
    global ocr_instance, doc_pipeline

    if use_gpu:
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError("PaddlePaddle CUDA support is required for GPU mode.")
        paddle.set_device("gpu")
        device = paddle.get_device()
    else:
        paddle.set_device("cpu")
        device = "cpu"

    # Text OCR engine
    ocr_instance = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=lang,
    )

    # Document structure pipeline
    doc_pipeline = PPStructureV3(
        use_doc_orientation_classify=use_orientation,
        use_doc_unwarping=use_unwarp,
        use_textline_orientation=use_textline,
        use_chart_recognition=use_chart,
        device="gpu" if use_gpu else "cpu",
    )

    return f"Pipelines initialized (Device: {device}, Lang: {lang})"


# ==============================
# Utilities
# ==============================
def list_pdf_files(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return "Invalid folder path", []

    pdf_files = list(Path(folder_path).rglob("*.pdf"))
    if not pdf_files:
        return "No PDF files found", []

    text = "Found PDF files:\n"
    for i, f in enumerate(pdf_files, 1):
        text += f"{i}. {f}\n"

    return text, [str(f) for f in pdf_files]


def is_readable_text(text: str) -> bool:
    if not text or len(text) < 3:
        return False

    letters = sum(c.isalpha() for c in text)
    spaces = sum(c.isspace() for c in text)
    weird = sum(not (c.isalnum() or c.isspace() or c in ".,;:!?()-[]{}\"'") for c in text)

    if letters < len(text) * 0.4:
        return False
    if weird > len(text) * 0.1:
        return False
    if len(text) > 20 and spaces == 0:
        return False

    return True


# ==============================
# Core processing
# ==============================
def process_pdf(
    pdf_path: str,
    force_ocr: bool,
    render_scale: float,
    export_txt: bool,
    export_json: bool,
    export_md: bool,
    struct_save_dir: str,
    progress=gr.Progress(),
):
    """Process a single PDF with optional structured exports.

    Returns a dict with text, page stats, and paths used.
    """
    global ocr_instance, doc_pipeline

    pdf_name = Path(pdf_path).name
    result = {
        "filename": pdf_name,
        "total_pages": 0,
        "full_text": "",
        "pages": [],
    }

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        result["total_pages"] = total_pages

        all_text = []

        for page_num in range(total_pages):
            progress((page_num + 1) / max(1, total_pages), desc=f"{pdf_name} page {page_num + 1}/{total_pages}")

            page = doc[page_num]
            embedded = page.get_text().strip()

            # Render image for structure pipeline
            mat = fitz.Matrix(render_scale, render_scale)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(tempfile.gettempdir(), f"doc_{os.getpid()}_{page_num}.png")
            pix.save(img_path)

            page_text_collected = False
            page_text = None

            try:
                # Export structured artifacts if requested
                if export_json or export_md:
                    image = Image.open(img_path).convert("RGB")
                    img_np = np.array(image)
                    doc_out = doc_pipeline.predict(img_np)
                    for res in doc_out:
                        if export_json:
                            res.save_to_json(save_path=struct_save_dir)
                        if export_md:
                            res.save_to_markdown(save_path=struct_save_dir)

                # Plain text generation
                if export_txt:
                    if not force_ocr and embedded and len(embedded) > 50:
                        page_text = embedded
                        page_text_collected = True
                        result["pages"].append({"page": page_num + 1, "method": "embedded"})
                    else:
                        # OCR on the rendered image
                        image = Image.open(img_path).convert("RGB")
                        img_np = np.array(image)
                        try:
                            ocr_res = ocr_instance.predict(img_np)
                        except AttributeError:
                            ocr_res = ocr_instance.ocr(img_np)

                        lines = []
                        if ocr_res and ocr_res[0]:
                            for line in ocr_res[0]:
                                txt = line[1][0]
                                if is_readable_text(txt):
                                    lines.append(txt)

                        page_text = "\n".join(lines)
                        page_text_collected = bool(page_text)
                        result["pages"].append({"page": page_num + 1, "method": "ocr" if page_text_collected else "ocr_empty"})

                # Accumulate text
                if export_txt and page_text_collected:
                    all_text.append(page_text)

            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)

        doc.close()

        result["full_text"] = "\n\n".join(all_text)
        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "filename": pdf_name,
        }


def run_batch(
    folder_path: str,
    use_gpu: bool,
    lang: str,
    use_orientation: bool,
    use_unwarp: bool,
    use_textline: bool,
    use_chart: bool,
    force_ocr: bool,
    render_scale: float,
    export_txt: bool,
    export_json: bool,
    export_md: bool,
    progress=gr.Progress(),
):
    """Process all PDFs under folder_path. Writes text to ocr_results and saves structured outputs to doc_results.

    Returns (summary_text, text_output_root, struct_output_root)
    """
    global ocr_instance, doc_pipeline

    # Ensure pipelines are initialized
    if ocr_instance is None or doc_pipeline is None:
        initialize_pipelines(
            use_gpu=use_gpu,
            lang=lang,
            use_orientation=use_orientation,
            use_unwarp=use_unwarp,
            use_textline=use_textline,
            use_chart=use_chart,
        )

    root_dir = Path(folder_path).resolve()
    txt_root = root_dir / "ocr_results"
    struct_root = root_dir / "doc_results"
    txt_root.mkdir(exist_ok=True)
    struct_root.mkdir(exist_ok=True)

    _, pdf_files = list_pdf_files(folder_path)
    total = len(pdf_files)
    if total == 0:
        return "No PDF files found", str(txt_root), str(struct_root)

    summary = []

    for i, pdf in enumerate(pdf_files):
        pdf_path = Path(pdf).resolve()
        rel_path = pdf_path.relative_to(root_dir)

        # Text output paths
        out_file = txt_root / rel_path.parent / f"{rel_path.stem}_ocr.txt"
        err_file = txt_root / rel_path.parent / f"{rel_path.stem}_ERROR.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Structured output path (directory per page is created by saver under temp; we keep root dir here for reference)
        (struct_root / rel_path.parent).mkdir(parents=True, exist_ok=True)

        progress((i + 1) / total, desc=f"Processing {rel_path}")

        # Dedicated directory for structured exports per PDF
        pdf_struct_dir = struct_root / rel_path.parent / rel_path.stem
        pdf_struct_dir.mkdir(parents=True, exist_ok=True)

        res = process_pdf(
            str(pdf_path),
            force_ocr=force_ocr,
            render_scale=render_scale,
            export_txt=export_txt,
            export_json=export_json,
            export_md=export_md,
            struct_save_dir=str(pdf_struct_dir),
            progress=progress,
        )

        if "error" in res:
            err_file.write_text(
                f"ERROR: {res['error']}\n\n{res.get('traceback','')}",
                encoding="utf-8",
            )
            summary.append(f"ERROR: {rel_path}")
        else:
            if export_txt:
                out_file.write_text(res.get("full_text", ""), encoding="utf-8")
            summary.append(f"OK: {rel_path} ({res['total_pages']} pages)")

    return "\n".join(summary), str(txt_root), str(struct_root)


# ==============================
# Gradio UI
# ==============================
with gr.Blocks(title="PP-StructureV3 Document Parser + Batch OCR") as app:
    gr.Markdown("# Document Structure Parsing and Batch OCR (PaddleOCR)")

    with gr.Row():
        folder_input = gr.Textbox(label="Root PDF Folder", value="/workspace")
        use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=True)
        lang_dd = gr.Dropdown(
            label="Language",
            choices=[
                "en",
                "ch",
                "fr",
                "de",
                "es",
                "it",
                "pt",
                "ru",
                "ja",
                "ko",
            ],
            value="en",
        )

    gr.Markdown("## PP-StructureV3 Options")
    with gr.Row():
        opt_orientation = gr.Checkbox(label="Orientation classify", value=True)
        opt_unwarp = gr.Checkbox(label="Unwarp page", value=True)
        opt_textline = gr.Checkbox(label="Textline orientation", value=True)
        opt_chart = gr.Checkbox(label="Chart recognition", value=False)

    gr.Markdown("## Outputs and Performance")
    with gr.Row():
        export_txt = gr.Checkbox(label="Also export plain text (.txt)", value=True)
        export_json = gr.Checkbox(label="Export JSON (structured)", value=True)
        export_md = gr.Checkbox(label="Export Markdown (structured)", value=True)
        force_ocr = gr.Checkbox(label="Force OCR (ignore embedded text)", value=False)

    render_scale = gr.Slider(
        minimum=1.0, maximum=3.0, step=0.5, value=2.0, label="Render scale (PDF → image)"
    )

    init_btn = gr.Button("Initialize Pipelines")
    init_out = gr.Textbox()

    list_btn = gr.Button("List PDFs")
    list_out = gr.Textbox(lines=8)

    run_btn = gr.Button("Run Batch")
    results_output = gr.Textbox(lines=12, label="Summary")
    txt_output_folder = gr.Textbox(label="Text output root")
    struct_output_folder = gr.Textbox(label="Structured output root")

    init_btn.click(
        initialize_pipelines,
        [
            use_gpu_checkbox,
            lang_dd,
            opt_orientation,
            opt_unwarp,
            opt_textline,
            opt_chart,
        ],
        init_out,
    )

    list_btn.click(lambda p: list_pdf_files(p)[0], folder_input, list_out)

    run_btn.click(
        run_batch,
        [
            folder_input,
            use_gpu_checkbox,
            lang_dd,
            opt_orientation,
            opt_unwarp,
            opt_textline,
            opt_chart,
            force_ocr,
            render_scale,
            export_txt,
            export_json,
            export_md,
        ],
        [results_output, txt_output_folder, struct_output_folder],
    )


if __name__ == "__main__":
    print("Starting PP-StructureV3 Document Parser + Batch OCR UI...")
    app.launch(server_name="0.0.0.0", server_port=7889)
