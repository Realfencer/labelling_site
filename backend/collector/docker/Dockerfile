FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MINER_FORCE_CPU=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install numpy==2.5.2 opencv-python-headless==5.0.0.93 onnxruntime==1.28.0 onnx==1.22.0 pydantic==2.13.4 pillow==12.3.0 PyYAML==6.0.3 scipy requests huggingface_hub
COPY eval_runner.py dpp_runner.py /app/
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python", "/app/eval_runner.py"]
