## excuter

### op-mem-ompsimd

ompsimd是DeepX框架的cpu执行器进程

+ 采用OMP+SIMD加速tensor计算
+ double和float采用openblas加速
+ 使用了jemalloc内存池管理内存。


#### 1. 安装依赖

 安装apt依赖

```
sudo apt-get update
        sudo apt-get install -y \
          build-essential \
          cmake \
          libopenblas-dev \
          libyaml-cpp-dev \
          libjemalloc-dev \
          libgtest-dev \
          clang \
          git
```
 
源码依赖安装

```
sudo apt-get install -y libgtest-dev

# 克隆 Highway
git clone --depth 1 --branch ${HIGHWAY_VERSION} https://github.com/google/highway.git
cd highway
mkdir -p build && cd build

# 使用标准的 CMake 构建流程
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DHWY_SYSTEM_GTEST=ON \
    -DHWY_ENABLE_TESTS=OFF

# 构建和安装
make -j$(nproc)
sudo make install
sudo ldconfig  # 更新动态链接库缓存

# 确保头文件正确安装
sudo cp -r ../hwy /usr/local/include/
```

#### 2. 开发环境

c++ 17

