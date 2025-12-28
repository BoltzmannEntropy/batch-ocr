#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import importlib.util
import platform


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class DummyProgress:
    def __call__(self, *args, **kwargs):
        pass


def run_classic(root: Path, use_gpu: bool, force_ocr: bool, min_embedded_chars: int):
    app_py = Path(__file__).resolve().parent / 'app.py'
    mod = load_module(app_py, 'batch_ocr_app')

    print(f"Initializing OCR (classic) | use_gpu={use_gpu}")
    print(mod.initialize_ocr(use_gpu=use_gpu))

    summary, out_root = mod.ocr_all_pdfs(
        str(root), use_gpu=use_gpu, progress=DummyProgress(), force_ocr=force_ocr, min_embedded_chars=min_embedded_chars
    )
    print(summary)

    out_dir = Path(out_root)
    out_dir.mkdir(exist_ok=True)
    (out_dir / '_batch_summary.txt').write_text(summary, encoding='utf-8')
    print(f"Saved summary to: {out_dir / '_batch_summary.txt'}")


def run_structure(
    root: Path,
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
):
    app2_py = Path(__file__).resolve().parent / 'app2.py'
    mod = load_module(app2_py, 'batch_ocr_app2')

    print(
        f"Initializing PP-StructureV3 + OCR | use_gpu={use_gpu}, lang={lang}, "
        f"orientation={use_orientation}, unwarp={use_unwarp}, textline={use_textline}, chart={use_chart}"
    )
    print(
        mod.initialize_pipelines(
            use_gpu=use_gpu,
            lang=lang,
            use_orientation=use_orientation,
            use_unwarp=use_unwarp,
            use_textline=use_textline,
            use_chart=use_chart,
        )
    )

    summary, txt_root, struct_root = mod.run_batch(
        folder_path=str(root),
        use_gpu=use_gpu,
        lang=lang,
        use_orientation=use_orientation,
        use_unwarp=use_unwarp,
        use_textline=use_textline,
        use_chart=use_chart,
        force_ocr=force_ocr,
        render_scale=render_scale,
        export_txt=export_txt,
        export_json=export_json,
        export_md=export_md,
        progress=DummyProgress(),
    )

    print(summary)
    Path(txt_root).mkdir(exist_ok=True)
    Path(struct_root).mkdir(exist_ok=True)
    (Path(txt_root) / '_batch_summary.txt').write_text(summary, encoding='utf-8')
    print(f"Saved summary to: {Path(txt_root) / '_batch_summary.txt'}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Headless OCR runner for batch-ocr')
    parser.add_argument('--root', required=True, help='Root folder containing PDFs')
    parser.add_argument('--mode', choices=['classic', 'structure'], default='classic', help='Pipeline mode')

    # GPU flags with OS-aware default
    default_use_gpu = platform.system().lower() != 'darwin'
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--use-gpu', dest='use_gpu', action='store_true', help='Enable GPU if available')
    g.add_argument('--no-gpu', dest='use_gpu', action='store_false', help='Disable GPU (CPU only)')
    parser.set_defaults(use_gpu=default_use_gpu)

    # Shared options
    parser.add_argument('--force-ocr', action='store_true', default=False, help='Ignore embedded text and force OCR')
    parser.add_argument('--min-embedded-chars', type=int, default=300, help='Min chars/page to trust embedded text')

    # Structure-specific options
    parser.add_argument('--lang', default='en', help='Language code for OCR (structure mode)')
    parser.add_argument('--orientation', dest='use_orientation', action='store_true', default=True)
    parser.add_argument('--no-orientation', dest='use_orientation', action='store_false')
    parser.add_argument('--unwarp', dest='use_unwarp', action='store_true', default=True)
    parser.add_argument('--no-unwarp', dest='use_unwarp', action='store_false')
    parser.add_argument('--textline', dest='use_textline', action='store_true', default=True)
    parser.add_argument('--no-textline', dest='use_textline', action='store_false')
    parser.add_argument('--chart', dest='use_chart', action='store_true', default=False)
    parser.add_argument('--no-chart', dest='use_chart', action='store_false')
    # (force-ocr handled as shared flag)
    parser.add_argument('--render-scale', type=float, default=2.0, help='PDF render scale for structure mode')
    parser.add_argument('--export-txt', dest='export_txt', action='store_true', default=True)
    parser.add_argument('--no-export-txt', dest='export_txt', action='store_false')
    parser.add_argument('--export-json', dest='export_json', action='store_true', default=True)
    parser.add_argument('--no-export-json', dest='export_json', action='store_false')
    parser.add_argument('--export-md', dest='export_md', action='store_true', default=True)
    parser.add_argument('--no-export-md', dest='export_md', action='store_false')

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not root.exists():
        parser.error(f'Root does not exist: {root}')

    if args.mode == 'classic':
        run_classic(root, use_gpu=args.use_gpu, force_ocr=args.force_ocr, min_embedded_chars=args.min_embedded_chars)
    else:
        run_structure(
            root=root,
            use_gpu=args.use_gpu,
            lang=args.lang,
            use_orientation=args.use_orientation,
            use_unwarp=args.use_unwarp,
            use_textline=args.use_textline,
            use_chart=args.use_chart,
            force_ocr=args.force_ocr,
            render_scale=args.render_scale,
            export_txt=args.export_txt,
            export_json=args.export_json,
            export_md=args.export_md,
            )


if __name__ == '__main__':
    main()
