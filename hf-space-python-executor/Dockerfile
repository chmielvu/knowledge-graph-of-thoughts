FROM python:3.12-slim

# Install system dependencies for audio/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

# -------------------------------------------
# Step 1: Install PyTorch CPU-only (pinned to 2.5.1 for PyG compatibility)
# -------------------------------------------
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu

# -------------------------------------------
# Step 2: Install PyTorch Geometric + extensions (CPU)
# -------------------------------------------
RUN pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# -------------------------------------------
# Step 3: Install remaining requirements
# -------------------------------------------
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -------------------------------------------
# Step 4: Copy application
# -------------------------------------------
COPY --chown=user . /app

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "python_executor:app"]