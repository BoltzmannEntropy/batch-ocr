import gradio as gr
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import tempfile
import traceback
import numpy as np
import platform

from paddleocr import PaddleOCR
import paddle

# ==============================
# Global OCR instance
# ==============================
ocr_instance = None


def initialize_ocr(use_gpu=True):
    """Initialize PaddleOCR with stable classic OCR (no doc pipeline).

    - On macOS (Darwin), GPU is not supported; force CPU.
    - If GPU is requested but CUDA is unavailable, gracefully fall back to CPU.
    """
    global ocr_instance

    is_macos = platform.system().lower() == "darwin"
    requested_gpu = bool(use_gpu)

    # Auto-disable GPU on macOS
    if is_macos and requested_gpu:
        print("macOS detected; CUDA is not available. Forcing CPU mode.")
        requested_gpu = False

    print(f"Initializing PaddleOCR | GPU requested: {use_gpu} | macOS: {is_macos}")

    if requested_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
        device = paddle.get_device()
    else:
        if requested_gpu and not paddle.is_compiled_with_cuda():
            print("CUDA not available in this Paddle build. Falling back to CPU.")
        paddle.set_device("cpu")
        device = "cpu"

    ocr_instance = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )

    return f"PaddleOCR initialized successfully (Device: {device})"


# ==============================
# PDF Listing (recursive)
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


# ==============================
# Text Quality Filter
# ==============================
def is_readable_text(text):
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
# OCR Single PDF
# ==============================
def ocr_pdf_file(pdf_path, progress=gr.Progress()):
    global ocr_instance
    pdf_name = Path(pdf_path).name

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        all_text = []
        page_results = []

        for page_num in range(total_pages):
            progress((page_num + 1) / total_pages,
                     desc=f"{pdf_name} | Page {page_num + 1}/{total_pages}")

            page = doc[page_num]
            embedded = page.get_text().strip()

            if embedded and len(embedded) > 50:
                all_text.append(embedded)
                page_results.append({"page": page_num + 1, "method": "embedded"})
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_path = os.path.join(tempfile.gettempdir(), f"ocr_{os.getpid()}_{page_num}.png")
            pix.save(img_path)

            try:
                image = Image.open(img_path).convert("RGB")
                img_np = np.array(image)

                try:
                    result = ocr_instance.predict(img_np)
                except AttributeError:
                    result = ocr_instance.ocr(img_np)

                page_text = []
                if result and result[0]:
                    for line in result[0]:
                        txt = line[1][0]
                        if is_readable_text(txt):
                            page_text.append(txt)

                if page_text:
                    all_text.append("\n".join(page_text))
                    page_results.append({"page": page_num + 1, "method": "ocr"})
                else:
                    page_results.append({"page": page_num + 1, "method": "ocr_empty"})

            except Exception as e:
                page_results.append({"page": page_num + 1, "method": "error", "error": str(e)})
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)

        doc.close()

        return {
            "filename": pdf_name,
            "total_pages": total_pages,
            "full_text": "\n\n".join(all_text),
            "pages": page_results
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "filename": pdf_name
        }


# ==============================
# OCR ALL PDFs (recursive + mirrored output)
# ==============================
def ocr_all_pdfs(folder_path, use_gpu, progress=gr.Progress()):
    global ocr_instance

    if ocr_instance is None:
        initialize_ocr(use_gpu)

    root_dir = Path(folder_path).resolve()
    output_root = root_dir / "ocr_results"
    output_root.mkdir(exist_ok=True)

    _, pdf_files = list_pdf_files(folder_path)
    total = len(pdf_files)

    summary = []

    for i, pdf in enumerate(pdf_files):
        pdf_path = Path(pdf).resolve()
        rel_path = pdf_path.relative_to(root_dir)

        out_file = output_root / rel_path.parent / f"{rel_path.stem}_ocr.txt"
        err_file = output_root / rel_path.parent / f"{rel_path.stem}_ERROR.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        progress((i + 1) / total, desc=f"OCR {rel_path}")

        if out_file.exists():
            summary.append(f"SKIPPED: {rel_path}")
            continue

        result = ocr_pdf_file(str(pdf_path), progress)

        if "error" in result:
            err_file.write_text(
                f"ERROR: {result['error']}\n\n{result.get('traceback','')}",
                encoding="utf-8"
            )
            summary.append(f"ERROR: {rel_path}")
        else:
            out_file.write_text(result["full_text"], encoding="utf-8")
            summary.append(f"OK: {rel_path} ({result['total_pages']} pages)")

    return "\n".join(summary), str(output_root)


# ==============================
# Gradio UI
# ==============================
with gr.Blocks(title="PDF OCR with PaddleOCR") as app:
    gr.Markdown("# Recursive Batch PDF OCR (PaddleOCR)")

    folder_input = gr.Textbox(
        label="Root PDF Folder",
        value="/workspace",
    )

    # Default GPU checkbox based on OS (off on macOS)
    default_use_gpu = platform.system().lower() != "darwin"
    use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=default_use_gpu)

    init_btn = gr.Button("Initialize OCR")
    init_output = gr.Textbox()

    list_btn = gr.Button("List PDFs")
    list_output = gr.Textbox(lines=8)

    run_btn = gr.Button("OCR All PDFs")
    results_output = gr.Textbox(lines=12)
    output_folder = gr.Textbox()

    init_btn.click(initialize_ocr, use_gpu_checkbox, init_output)
    list_btn.click(lambda p: list_pdf_files(p)[0], folder_input, list_output)
    run_btn.click(ocr_all_pdfs, [folder_input, use_gpu_checkbox], [results_output, output_folder])


if __name__ == "__main__":
    print("Starting PDF OCR Gradio UI...")
    app.launch(server_name="0.0.0.0", server_port=7888)
