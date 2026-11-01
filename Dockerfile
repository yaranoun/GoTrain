# GoTrain MCP server over streamable HTTP (CPU). Results, sandbox, cache and state live in the /data volume.
#   docker build -t gotrain-mcp .
#   docker run --rm -p 8765:8765 -v "$PWD/workspace:/data/workspace:ro" -v gotrain-data:/data gotrain-mcp
# The server has no authentication: publish the port only on a trusted network.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY gotrain ./gotrain
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir . \
 && rm -rf /src

ENV GOTRAIN_HOME=/data \
    GOTRAIN_WORKSPACE=/data/workspace \
    GOTRAIN_MCP_WORKSPACE=/data/mcp/workspace \
    GOTRAIN_CACHE=/data/cache \
    GOTRAIN_STATE_DIR=/data/state \
    GOTRAIN_STORE_ROOT=/data/store \
    GEO_BENCH_DIR=/data/geobench \
    HF_HOME=/data/hf \
    TORCH_HOME=/data/torch
VOLUME /data
WORKDIR /data
EXPOSE 8765
ENTRYPOINT ["gotrain-mcp"]
CMD ["--http", "--host", "0.0.0.0", "--port", "8765"]
