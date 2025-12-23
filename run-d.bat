@echo off
REM Run Batch OCR container and start the Gradio app

docker run --gpus all --rm -it ^
    --shm-size=8gb ^
    --ipc=host ^
    --ulimit memlock=-1 ^
    --ulimit stack=67108864 ^
    -p 7888:7888 ^
    -v "%cd%":/workspace ^
    -v "%USERPROFILE%/.cache":/root/.cache ^
    batch-ocr:latest ^
    bash -lc "python /workspace/app.py"


@REM If the container is built as a base without application code,
@REM you can also run an interactive shell:
@REM docker run --gpus all --rm -it -p 7888:7888 -v "%cd%":/workspace batch-ocr:latest bash
