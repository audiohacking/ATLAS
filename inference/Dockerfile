# GGML_BACKEND selects the GPU/compute back-end for Linux container builds:
#   cuda  (default) — NVIDIA CUDA via nvidia/cuda base image
#   rocm             — AMD ROCm/HIP via rocm/dev-ubuntu base image
#   cpu              — CPU-only fallback, no GPU required
#
# Apple Metal requires native macOS toolchain — use the GitHub Actions workflow
# (.github/workflows/build-inference.yml) which runs on macos-latest.
#
# CUDA_ARCHITECTURES targets:
#   89-real  — Ada Lovelace (RTX 4000 series, L40)
#   90-real  — Hopper       (H100, H200)
#   120-real — Blackwell    (GB200, RTX 5000 series)
# Narrow to your GPU to reduce compile time, e.g.:
#   docker build --build-arg CUDA_ARCHITECTURES=89-real .
ARG GGML_BACKEND=cuda
ARG CUDA_ARCHITECTURES=89-real;90-real;120-real

# === Build-stage base images (only the selected one is resolved) ===
FROM docker.io/nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base-cuda
FROM docker.io/rocm/dev-ubuntu-24.04:6.3-complete    AS base-rocm
FROM ubuntu:24.04                                     AS base-cpu

# === Builder ===
FROM base-${GGML_BACKEND} AS builder
ARG GGML_BACKEND=cuda
ARG CUDA_ARCHITECTURES=89-real;90-real;120-real

ENV DEBIAN_FRONTEND=noninteractive TMPDIR=/llama.cpp/tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggml-org/llama.cpp /llama.cpp \
    && mkdir -p /llama.cpp/tmp \
    && cd /llama.cpp \
    && case "$GGML_BACKEND" in \
         cuda) CMAKE_EXTRA="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" ;; \
         rocm) CMAKE_EXTRA="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906;gfx908;gfx90a;gfx1030;gfx1100" ;; \
         *)    CMAKE_EXTRA="" ;; \
       esac \
    && cmake -B build -DBUILD_SHARED_LIBS=OFF ${CMAKE_EXTRA} \
    && cmake --build build --config Release -j$(nproc)

# === Runtime base images ===
FROM docker.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime-cuda
FROM docker.io/rocm/dev-ubuntu-24.04:6.3-complete     AS runtime-rocm
FROM ubuntu:24.04                                      AS runtime-cpu

FROM runtime-${GGML_BACKEND} AS final
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /llama.cpp/build/bin/llama-cli    /usr/local/bin/
RUN mkdir -p /models /templates
COPY templates/ /templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
