# heap-plat 开发指南

> 开发 heap-metal (macOS 统一内存)。heap-cuda / heap-cpu 暂不开发。
> Redis 连通测试后，最先开发此组件——先有内存，才能计算。

## 1. 角色与职责

**heap-\* 的进程维持着 deepx 元程的堆在 \* 设备平台的高可用。**

heap-plat 管理 tensor 对象的生命周期：创建、删除、克隆。

| 能力 | 说明 |
|------|------|
| Tensor 创建 (newtensor) | 分配 POSIX shm + GPU buffer，写入元信息到 Redis |
| Tensor 删除 (deltensor) | 释放 shm，删除 Redis key |
| Tensor 克隆 (clonetensor) | 在指定设备上创建 tensor 副本 |
| 进程注册 | 启动时向 `/sys/heap-plat/` 注册 |

## 2. 当前状态

```
executor/heap-metal/
├── src/
│   ├── main.mm                   入口 (已连接 Redis, 有主循环)
│   └── lifecycle/
│       └── lifecycle.h           命令处理 (创建/删除/查询 tensor)

依赖 common-metal:
  src/shm_tensor.h/.mm            POSIX shm 创建/打开/关闭
  src/registry.h                  注册表抽象基类

代码量: 426 行
```

## 3. 通信模型

```
          VM                           heap-metal
          ──                           ─────────────
          PUSH cmd:heap-metal:0  ──→ RPOP/BLPOP 消费
                                          │
                                      分配/释放 shm
                                      写入 Redis 元信息
                                          │
          BLPOP done:<vtid>  ←────────── LPUSH 完成事件
```

## 4. 待开发任务

### 任务 H1: Redis 命令消费循环 (main.mm)

```cpp
// 伪代码
while (true) {
    auto cmd = redis.blpop("cmd:heap-metal:0", timeout_sec=5);
    if (!cmd) continue;  // 超时，下一轮

    auto req = json::parse(cmd);
    string vtid = req["vtid"];
    string pc   = req["pc"];
    string op   = req["op"];

    if (op == "newtensor") {
        handle_newtensor(req);
    } else if (op == "deltensor") {
        handle_deltensor(req);
    } else if (op == "clonetensor") {
        handle_clonetensor(req);
    }

    // 回复完成
    json done = {{"pc", pc}, {"status", "ok"}};
    redis.lpush("done:" + vtid, done.dump());
}
```

**依赖:** hiredis (Redis C 客户端)。Mac 上 `brew install hiredis`。

### 任务 H2: newtensor 实现

```
输入: {vtid, pc, op:"newtensor", key:"/models/weights", dtype:"f32", shape:[1024,512], device:"gpu0"}

处理:
  1. 计算 byte_size = element_count(shape) × dtype_size(dtype)
     例如: 1024×512×4 = 2,097,152 bytes (f32=4bytes)
  2. 生成 shm_name = "/deepx_t_" + random_hex(8)
  3. 调用 shm_tensor_create(shm_name, byte_size) → 分配 POSIX shm
  4. 构造 tensor 元信息 → SET /models/weights
  5. 回复 LPUSH done:<vtid> {"pc":"...", "status":"ok"}
```

**写入 Redis 的 tensor 元信息:**
```json
{
    "dtype": "f32",
    "shape": [1024, 512],
    "byte_size": 2097152,
    "device": "gpu0",
    "address": {
        "node": "n1",
        "type": "shm",
        "shm_name": "/deepx_t_a1b2c3d4"
    },
    "ctime": 1714000000,
    "version": 1
}
```

**Mac 统一内存说明:**
Mac Apple Silicon 上 CPU 和 GPU 共享物理内存。shm_open + mmap 返回的指针
可以直接被 Metal 使用（通过 newBufferWithBytesNoCopy 包装）。

### 任务 H3: deltensor 实现

```
输入: {vtid, pc, op:"deltensor", key:"/models/weights"}

处理:
  1. GET /models/weights → 获取 shm_name
  2. shm_tensor_unlink(shm_name) → 释放 POSIX shm
  3. UNLINK /models/weights → 删除 Redis key
  4. 回复 LPUSH done:<vtid> {"pc":"...", "status":"ok"}
```

### 任务 H4: clonetensor 实现

```
输入: {vtid, pc, op:"clonetensor", src:"/models/weights", dst:"/models/weights_gpu1", device:"gpu1"}

处理:
  1. GET src → 获取源 tensor 元信息
  2. 分配新 shm → shm_tensor_create(new_shm_name, src.byte_size)
  3. memcpy(src_ptr, dst_ptr, src.byte_size)  // Mac 统一内存下直接 memcpy
  4. SET dst → 新 tensor 元信息 (含 new_shm_name, device)
  5. 回复 LPUSH done:<vtid> {"pc":"...", "status":"ok"}
```

### 任务 H5: 进程注册

启动时注册到 Redis：

```
SET /sys/heap-plat/metal:0 = {
    "program": "heap-metal",
    "device": "gpu0",
    "status": "running",
    "pid": <getpid()>,
    "started_at": <timestamp>
}
```

退出时清理:
```
DELETE /sys/heap-plat/metal:0
```

## 5. 编译与运行 (macOS)

```bash
# 安装依赖
brew install hiredis

# 构建
cd executor/heap-metal
mkdir -p build && cd build
cmake .. && make

# 运行
./heap_metal
```

## 6. 验证方法

```bash
# 终端1: 启动 heap-metal
./heap_metal

# 终端2: 通过 redis-cli 发送测试命令
redis-cli RPUSH cmd:heap-metal:0 '{"vtid":"test1","pc":"[0,0]","op":"newtensor","key":"/test/x","dtype":"f32","shape":[100,100],"device":"gpu0"}'

# 查看结果
redis-cli GET /test/x
# 应返回: {"dtype":"f32","shape":[100,100],"byte_size":40000,"device":"gpu0","address":{...}}

# 检查完成通知
redis-cli BLPOP done:test1 1
# 应返回: {"pc":"[0,0]","status":"ok"}

# 删除
redis-cli RPUSH cmd:heap-metal:0 '{"vtid":"test1","pc":"[1,0]","op":"deltensor","key":"/test/x"}'
```

## 7. 开发量评估

| 任务 | 新增代码 | 难度 |
|------|---------|------|
| H1: 命令消费循环 | ~80 行 | 低 |
| H2: newtensor | ~80 行 | 低 |
| H3: deltensor | ~50 行 | 低 |
| H4: clonetensor | ~50 行 | 低 |
| H5: 进程注册 | ~40 行 | 低 |
| **合计** | **~300 行** | **低** |
