FROM docker.array2d.com/library/ubuntu:18.04

# 基础构建环境
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libyaml-cpp-dev \
    libjemalloc-dev \
    clang \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Highway SIMD 库

RUN mkdir -p thirdlib && \
    cd thirdlib && \
    git clone https://github.com/google/highway.git && \
    cd highway && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    cd ../../ && rm -rf thirdlib/highway

ADD cpp-common cpp-common
ADD op-mem-ompsimd op-mem-ompsimd
WORKDIR /home/op-mem-ompsimd

RUN rm -rf build && mkdir build && cd build && \
    cmake ..&& \
    make -j$(nproc)

CMD ["./build/bin/deepx-excuter-ompsimd"]