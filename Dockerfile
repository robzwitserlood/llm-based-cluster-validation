FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 via deadsnakes PPA and system tools.
# gcc-12/g++-12 are pinned as the nvcc host compiler via CUDAHOSTCXX below:
# CUDA 12.x only supports GCC ≤ 12, and the system default may be newer.
RUN apt-get update \
    && apt-get install -y software-properties-common curl git build-essential gcc-12 g++-12 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Tell nvcc to use GCC 12 as the host compiler for FlashInfer JIT compilation.
ENV CUDAHOSTCXX=/usr/bin/g++-12

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
COPY cluster_validator/ ./cluster_validator/

# Install all dependencies (including SGLang with GPU support)
RUN uv sync

# Copy the rest of the project
COPY config/ ./config/
COPY pipeline/ ./pipeline/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY Snakefile ./Snakefile
COPY .env.example ./.env.example

# SGLang inference server and MLflow model serve
EXPOSE 30000 5000

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-c", "\
print('LLM-based Cluster Validation — Snakemake targets:'); \
print('  uv run snakemake --dry-run all           # inspect the DAG'); \
print('  uv run snakemake all -j1                 # full pipeline end-to-end'); \
print('  uv run snakemake evaluate -j1            # baseline only'); \
print('  uv run snakemake optimize -j1            # BootstrapFewShot'); \
print('  uv run snakemake optimize_gepa -j1       # GEPA (needs ANTHROPIC_API_KEY)'); \
print('  uv run snakemake optimize_finetune -j1   # BootstrapFinetune (needs ANTHROPIC_API_KEY)'); \
print('  uv run snakemake deploy -j1              # log fine-tuned program to MLflow'); \
print(); \
print('Run with GPU support:'); \
print('  docker run --gpus all -e ANTHROPIC_API_KEY=<key> <image> uv run snakemake <target> -j1'); \
print(); \
print('SGLang is started/stopped automatically by the Snakefile rules — do not launch it manually.'); \
"]
