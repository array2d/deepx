# heap-metal

> macOS Metal 统一内存的 heap-plat 实现。
> Apple Silicon 上 CPU/GPU 共享物理内存，shm_open + mmap 的指针
> 可直接通过 newBufferWithBytesNoCopy 被 GPU 访问。
>
> **heap-metal 的进程维持着 deepx 元程的堆在 Metal 设备平台的高可用。**

## 1. 平台特性

| 特性 | 说明 |
|------|------|
| 设备 | Apple Silicon GPU (Metal) |
| 内存模型 | 统一内存 (CPU/GPU 共享物理内存) |
| 分配方式 | POSIX shm_open + ftruncate + mmap |
| shm 命名 | `/deepx_t_<8位随机hex>` |
| GPU 访问 | `newBufferWithBytesNoCopy` 直接包装 mmap 指针 |

## 2. 代码位置

```
executor/heap-metal/
├── CMakeLists.txt
├── src/
│   ├── main.mm                   入口: Redis 连接, 主循环
│   └── lifecycle/
│       └── lifecycle.h/.mm       newtensor / deltensor / clonetensor

依赖:
  executor/common-metal/
    ├── shm_tensor.h/.mm          POSIX shm 封装
    ├── registry.h/.mm            注册表抽象
    └── metal_device.h/.mm        Metal 设备管理
```

## 3. newtensor 流程

```
输入:
  {"vtid":"1", "pc":"[0,0]", "op":"newtensor", "key":"/models/W",
   "dtype":"f32", "shape":[1024,512], "device":"gpu0"}

步骤:
  1. 计算 byte_size = 1024 × 512 × 4 = 2,097,152
  2. 生成 shm_name = "/deepx_t_" + random_hex(8)
  3. shm_tensor_create(shm_name, byte_size):
     a) shm_open(shm_name, O_CREAT|O_RDWR, 0600) → fd
     b) ftruncate(fd, byte_size)
     c) mmap(NULL, byte_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0) → ptr
     d) close(fd)
  4. 构造 tensor 元信息 → SET /models/W
  5. LPUSH done:<vtid> {"pc":"[0,0]", "status":"ok"}
```

**Mac 统一内存优势:**
mmap 返回的 CPU 指针在 Apple Silicon 上可直接被 GPU 使用，
无需 `cudaMemcpy` 或显式数据传输。

```objc
// op-metal 侧使用示例
void* cpu_ptr = mmap_result;
id<MTLBuffer> gpu_buf = [device newBufferWithBytesNoCopy:cpu_ptr
                                                 length:byte_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
```

## 4. deltensor 流程

```
输入:
  {"vtid":"1", "pc":"[5,0]", "op":"deltensor", "key":"/models/W"}

步骤:
  1. GET /models/W → 获取 address.shm_name
  2. shm_unlink(shm_name) → 标记删除
  3. munmap(ptr, byte_size) → 解除映射 (如果有缓存)
  4. UNLINK /models/W → 删除 Redis key
  5. LPUSH done:<vtid> {"pc":"[5,0]", "status":"ok"}
```

## 5. clonetensor 流程

```
输入:
  {"vtid":"1", "pc":"[0,0]", "op":"clonetensor",
   "src":"/models/W", "dst":"/models/W_gpu1", "device":"gpu1"}

步骤:
  1. GET /models/W → 源 tensor 元信息
  2. 调用 newtensor 逻辑创建目标 tensor (新 shm_name, device=gpu1)
  3. shm_open(src_shm_name) → 映射源数据
  4. memcpy(dst_ptr, src_ptr, byte_size)  // 统一内存下直接拷贝
  5. SET /models/W_gpu1 → 新 tensor 元信息
  6. LPUSH done:<vtid> ...
```

## 6. 进程注册

```
启动时:
  SET /sys/heap-plat/metal:0 = {
      "program": "heap-metal",
      "device": "gpu0",
      "status": "running",
      "pid": <getpid()>,
      "started_at": <unix_timestamp>
  }

退出时:
  DELETE /sys/heap-plat/metal:0
```

## 7. Redis 命令队列

| Key | 说明 |
|-----|------|
| `cmd:heap-metal:0` | 默认实例的命令队列 (监听) |
| `cmd:heap-metal:1` | 第 2 个实例 (如有) |
| `done:<vtid>` | 完成通知 (写入) |

## 8. 依赖

```bash
brew install hiredis    # Redis C 客户端
```

CMakeLists.txt 需链接:
- hiredis
- common-metal (shm_tensor, metal_device)

## 9. 测试

```bash
# 终端1: heap-metal
./heap_metal

# 终端2: redis-cli 模拟
redis-cli RPUSH cmd:heap-metal:0 \
  '{"vtid":"t","pc":"[0,0]","op":"newtensor","key":"/test/x","dtype":"f32","shape":[10,10],"device":"gpu0"}'

redis-cli GET /test/x
# → {"dtype":"f32","shape":[10,10],"byte_size":400,"device":"gpu0","address":{"type":"shm","shm_name":"/deepx_t_a1b2c3d4",...}}

redis-cli BLPOP done:t 1

redis-cli RPUSH cmd:heap-metal:0 \
  '{"vtid":"t","pc":"[1,0]","op":"deltensor","key":"/test/x"}'

redis-cli GET /test/x
# → (nil)
```

## 10. 开发量

~300 行 C++/ObjC 新增代码。当前已有 426 行基础设施。
