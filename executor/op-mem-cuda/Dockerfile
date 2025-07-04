FROM docker.array2d.com/nvidia/cuda:12.8.0-devel-ubuntu22.04


# 基础构建环境
RUN apt-get update && apt-get install -y \
    cmake \
    libyaml-cpp-dev \
    clang \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /home/
# CUDA 已经预装，无需额外配置
ADD cpp-common cpp-common
ADD op-mem-cuda op-mem-cuda
WORKDIR /home/op-mem-cuda

RUN rm -rf build && mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

CMD ["./build/bin/deepx-excuter-cuda"]