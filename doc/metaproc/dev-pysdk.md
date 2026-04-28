# pysdk 开发指南

> pysdk 是 DeepX 的 Python 算法前端，负责注册源码到 Redis、创建 vthread。
> 当前 front/py/deepx 已有 9,524 行代码，需要在现有基础上增量改造。

## 1. 角色与职责

pysdk 是用户侧接口，负责将用户的模型代码翻译为 `/src/func/` 下的函数定义和 `/vthread/` 下的执行单元。

| 能力 | 说明 |
|------|------|
| 函数注册 | 将 func 签名 + 指令序列写入 `/src/func/<name>/` |
| Vthread 创建 | 分配 vtid，写入入口 CALL 指令，唤醒 VM |
| Tensor 管理 | 创建/删除堆 tensor，通过 heap-plat |
| 状态查询 | GET `/vthread/<vtid>` 检查执行状态和结果 |
| Func 缓存 | 避免重复发送未变化的函数 |

## 2. 当前状态

```
front/py/deepx/
├── tensor/             tensor 操作 (elementwise, matmul, reduce, ...)
├── nn/                 deepxIR 生成, parameter
├── optim/              sgd, adam
└── ...

代码量: 9,524 行 Python，约 30 个文件
```

当前模式: 通过 HTTP 直接发送 deepxIR 序列到 Redis HTTP 代理。
改造后: 通过 KVSpace 客户端写入 `/src/func/` 和 `/vthread/`。

## 3. 改造策略

**增量改造** — 不重写现有 9,524 行代码，而是:

1. 新增 `KVSpace` 客户端类 (封装 Redis 操作)
2. 新增 `FuncCache` 类 (避免重复发送)
3. 新增 `VThreadCreator` 类 (创建 vthread)
4. 现有 `tensor/` 算子逐步适配，从 HTTP 模式切换到 KVSpace 模式

## 4. 待开发任务

### 任务 P1: KVSpace 客户端抽象

封装 Redis 操作为 KVSpace 语义接口:

```python
import redis
import json

class KVSpace:
    """
    KV 空间客户端，封装 Redis 操作为元程路径语义。

    用法:
        kv = KVSpace("redis://localhost:6379")
        kv.set("/models/weights", {"dtype": "f32", "shape": [1024, 512]})
        meta = kv.get("/models/weights")
    """

    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    # === 基本 KV 操作 ===
    def get(self, key: str):
        val = self.redis.get(key)
        return json.loads(val) if val else None

    def set(self, key: str, value):
        return self.redis.set(key, json.dumps(value))

    def delete(self, key: str):
        return self.redis.delete(key)

    def exists(self, key: str) -> bool:
        return self.redis.exists(key) > 0

    def keys(self, pattern: str) -> list:
        return self.redis.keys(pattern)

    # === 函数源码写入 ===
    def set_func(self, name: str, signature: str, instructions: list):
        """
        将函数定义写入 /src/func/<name>/

        Args:
            name: 函数名, 如 "gemm", "forward"
            signature: dxlang 签名, 如 "(gemm(A:tensor<f32>, B:tensor<f32>) -> (Y:tensor<f32>))"
            instructions: 指令列表, 每条是 dxlang 字符串
                如 ["matmul(A, B) -> ./Y", "relu(./Y) -> ./out"]
        """
        self.set(f"/src/func/{name}", signature)
        for i, inst in enumerate(instructions):
            self.set(f"/src/func/{name}/{i}", inst)

    def get_func(self, name: str) -> dict:
        """读取函数定义 (调试用)"""
        sig = self.get(f"/src/func/{name}")
        insts = []
        i = 0
        while True:
            inst = self.get(f"/src/func/{name}/{i}")
            if inst is None:
                break
            insts.append(inst)
            i += 1
        return {"signature": sig, "instructions": insts}

    # === Vthread 管理 ===
    def alloc_vtid(self) -> str:
        """分配新的 vthread ID"""
        return str(self.redis.incr("/sys/vtid_counter"))

    def get_vthread(self, vtid: str) -> dict:
        """获取 vthread 状态"""
        return self.get(f"/vthread/{vtid}")

    def wait_vthread(self, vtid: str, timeout: float = 30.0,
                     poll_interval: float = 0.05) -> dict:
        """
        等待 vthread 执行完成

        Returns:
            vthread 最终状态 (status 为 "done" 或 "error")
        """
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_vthread(vtid)
            if state is None:
                raise RuntimeError(f"vthread {vtid} not found")
            if state["status"] in ("done", "error"):
                return state
            time.sleep(poll_interval)
        raise TimeoutError(f"vthread {vtid} timeout after {timeout}s")

    # === 命令队列 ===
    def push(self, queue: str, value):
        return self.redis.rpush(queue, json.dumps(value))

    def pop(self, queue: str, timeout: int = 0):
        result = self.redis.blpop(queue, timeout)
        return json.loads(result[1]) if result else None
```

### 任务 P2: FuncCache

避免重复发送相同 func 定义:

```python
class FuncCache:
    """函数缓存，基于内容 hash 避免重复发送"""

    def __init__(self, kv: KVSpace):
        self.kv = kv
        self._hashes: dict = {}

    @staticmethod
    def _compute_hash(signature: str, instructions: list) -> str:
        import hashlib
        content = signature + "".join(instructions)
        return hashlib.md5(content.encode()).hexdigest()

    def set_if_changed(self, name: str, signature: str, instructions: list) -> bool:
        """
        仅在函数内容变化时才写入 KV 空间

        Returns:
            True 如果有变更并写入, False 如果未变化跳过
        """
        h = self._compute_hash(signature, instructions)
        if self._hashes.get(name) == h:
            return False
        self.kv.set_func(name, signature, instructions)
        self._hashes[name] = h
        return True

    def invalidate(self, name: str):
        """强制下次写入 (用于调试)"""
        self._hashes.pop(name, None)
```

### 任务 P3: Vthread 创建器

```python
class VThreadCreator:
    def __init__(self, kv: KVSpace):
        self.kv = kv

    def create(self, entry_func: str, bindings: dict,
               entry_inst: str = "[0,0]") -> str:
        """
        创建 vthread 并通知 VM

        写入 /vthread/<vtid>/ 的入口 CALL 指令:
          [0, 0] = "call"
          [0,-1] = entry_func
          [0,-2] = bindings[绑定1]
          ...
          [0, 1] = "./out"  (默认返回值)

        Args:
            entry_func: 入口函数名, 如 "forward"
            bindings: 参数绑定, 如 {"A": "./a", "B": "./b", "alpha": 1.0}
            entry_inst: 入口指令坐标, 默认 "[0,0]"

        Returns:
            vtid 字符串
        """
        vtid = self.kv.alloc_vtid()

        # 设置 vthread 状态
        self.kv.set(f"/vthread/{vtid}", {
            "pc": entry_inst,
            "status": "init"
        })

        # 写入入口 CALL 指令
        base = f"/vthread/{vtid}/{entry_inst}"
        self.kv.set(base, "call")                           # [0,0] = opcode
        self.kv.set(f"{base}/-1", entry_func)               # [0,-1] = func_name

        # 写入实参: [0,-2], [0,-3], ...
        for i, (param_name, param_value) in enumerate(bindings.items()):
            self.kv.set(f"{base}/{-i-2}", str(param_value))

        # 默认返回值槽位
        self.kv.set(f"{base}/1", "./out")                   # [0,1] = 返回值

        # 唤醒 VM
        self.kv.push("notify:vm", {"event": "new_vthread", "vtid": vtid})

        return vtid
```

### 任务 P4: 现有算子适配

当前 `tensor/elementwise.py` 等文件直接发送 HTTP 请求。
改造为通过 KVSpace 写入 `/src/func/`:

```python
# === 改造前 (当前模式) ===
def add(A, B) -> C:
    # 通过 HTTP 发送 deepxIR
    send_ir({"op": "add", "A": A, "B": B, "C": C})

# === 改造后 (KVSpace 模式) ===
def add(kv: KVSpace, A: str, B: str, out_name: str):
    """
    注册 add 函数源码到 KV 空间

    Args:
        kv: KVSpace 客户端
        A: 输入 tensor 的路径 (如 "./a", "/models/X")
        B: 输入 tensor 的路径
        out_name: 输出变量名 (如 "c")
    """
    func_name = f"add_{A.replace('/', '_')}_{B.replace('/', '_')}"
    signature = f"(add(A:tensor, B:tensor) -> ({out_name}:tensor))"
    instructions = [f"add({A}, {B}) -> ./{out_name}"]
    kv.set_func(func_name, signature, instructions)
    return func_name
```

**适配策略:** 先不改现有业务代码，新增一个 `kv_adapter.py` 模块，
提供与原接口兼容的适配函数。后续逐步将业务逻辑迁移到新接口。

### 任务 P5: Tensor 元信息辅助

```python
def tensor_meta(dtype: str, shape: list, device: str = "gpu0") -> dict:
    """构造 tensor 元信息 (用于 newtensor 请求)"""
    dtype_sizes = {
        "f16": 2, "f32": 4, "f64": 8, "bf16": 2,
        "i8": 1, "i16": 2, "i32": 4, "i64": 8, "u8": 1
    }
    import math
    count = math.prod(shape)
    byte_size = count * dtype_sizes[dtype]
    return {
        "dtype": dtype,
        "shape": shape,
        "byte_size": byte_size,
        "device": device
    }
```

## 5. 依赖

```bash
pip install redis
```

## 6. 验证方法

```python
# test_kvspace.py
from deepx.kvspace import KVSpace, FuncCache, VThreadCreator

def test_kvspace():
    kv = KVSpace("redis://localhost:6379")

    # 1. 写入函数源码
    kv.set_func("add_test",
        signature="(add_test(A:tensor, B:tensor) -> (C:tensor))",
        instructions=["add(A, B) -> C"]
    )

    # 2. 验证写入
    func = kv.get_func("add_test")
    assert func["signature"] == "(add_test(A:tensor, B:tensor) -> (C:tensor))"
    assert func["instructions"][0] == "add(A, B) -> C"

    # 3. FuncCache 测试
    cache = FuncCache(kv)
    assert cache.set_if_changed("add_test", func["signature"], func["instructions"]) == True
    # 第二次应跳过
    assert cache.set_if_changed("add_test", func["signature"], func["instructions"]) == False

    # 4. 创建 vthread (需要 VM 运行)
    creator = VThreadCreator(kv)
    kv.set("/vthread/1/a", "tensor_ref_a")  # 模拟已有局部变量
    kv.set("/vthread/1/b", "tensor_ref_b")
    vtid = creator.create("add_test", {"A": "./a", "B": "./b"})
    print(f"Created vthread: {vtid}")

    # 5. 等待执行 (VM 需运行)
    state = kv.wait_vthread(vtid, timeout=5)
    print(f"Vthread state: {state}")
```

## 7. 开发量评估

| 任务 | 新增代码 | 难度 |
|------|---------|------|
| P1: KVSpace 客户端 (~150 行 Python) | ~150 行 | 低 |
| P2: FuncCache (~40 行) | ~40 行 | 低 |
| P3: Vthread 创建器 (~60 行) | ~60 行 | 低 |
| P4: 现有算子适配 (kv_adapter 模块) | ~150 行 | 中 |
| P5: Tensor 元信息辅助 (~30 行) | ~30 行 | 低 |
| 现有 9,524 行业务代码逐步迁移 | 待评估 | 中 |
| **合计 (新增)** | **~500 行 Python** | **低** |

## 8. 注意

1. **pysdk 是增量改造** — 不重写现有代码，先加 KVSpace 层，逐步迁移
2. **FuncCache 关键** — 避免每轮训练重复发送相同 func 定义
3. **Vthread 创建后通知 VM** — 通过 `notify:vm` 队列
4. **编译阶段暂跳过** — 当前 pysdk 直接写 `/src/func/`，VM 在读时做 eager 展开。未来编译器阶段再引入 `/op/<backend>/func/`
