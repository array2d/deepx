# skill: add-metal-kernel → 新增 Metal GPU kernel

引导式工作流，按步骤新增一个完整的 Metal GPU 算子。

## 前置条件

- 本地有 Xcode + Metal 工具链
- 已运行过 `/build-exop-metal` 确认构建环境正常
- 了解 exop-metal 代码结构 (参考 `@dev-exop-metal`)

## 工作流步骤

### 1. 确认算子范围

回答以下问题:
- 算子名称? (e.g., "gelu", "softmax")
- 几元操作? 一元 / 二元 / 多元?
- 支持哪些 dtype? f32 必须，扩展 f16/i8/i16/i32/i64?
- 存在 CPU fallback 需求吗?

### 2. 写 Metal Shader

编辑: `executor/exop-metal/src/deepx/tensorfunc/elementwise_miaobyte.metal`

模板 (一元):
```metal
kernel void newop_f32(device const float* X [[buffer(0)]],
                       device float*       Y [[buffer(1)]],
                       constant uint&      n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid < n) { Y[gid] = /* 实现 */; }
}
```

模板 (二元):
```metal
kernel void newop_f32(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float*       C [[buffer(2)]],
                       constant uint&      n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid < n) { C[gid] = /* 实现 */; }
}
```

生成所有 dtype 变体: f16, f32, i8, i16, i32, i64
整数 kernel 注意显式 cast: `(char)(...)`

### 3. 写 Host 桥接函数

编辑: `executor/exop-metal/src/deepx/tensorfunc/elementwise_miaobyte.hpp`

参考已有函数模式 (如 `add_f32` → `relu_f32`):
- `extern bool newop_f32(const float* x, float* y, int64_t n);`
- 实现: 获取 MetalContext → 加载 library → 创建 pipeline → 分配 buffer → dispatch → 等待完成
- 检查 commandBuffer.error → 返回成功/失败

### 4. 注册算子 + 分派

编辑: `executor/exop-metal/src/client/main.cpp`

**注册**: 在 `register_instance()` 中:
```cpp
redis_cmd(c, "RPUSH %s %s", "/op/exop-metal/list", "newop");
```

**分派**: 在 `execute_task()` 中增加分支:
```cpp
// 如果是已有一元分派函数 → 加到 dispatch_unary 的 if 条件
else if (opcode == "newop" && input_ptrs.size() == 1) {
    ok = dispatch_unary(opcode, dtype, input_ptrs[0], out_shm.addr, n);
    if (!ok) error = "Metal kernel dispatch failed for newop:" + dtype;
}
```

在 `dispatch_unary()` 中增加:
```cpp
if (opcode == "newop") {
    if (dtype == "f32" || dtype == "float32") return newop_f32(...);
    // ... 其他 dtype
}
```

### 5. TfFactory 注册 (可选，调度器用)

编辑: `executor/exop-metal/src/deepx/tf/register_miaobyte.hpp`

```cpp
factory.add_tf(std::make_shared<NewOp<Author>>(
    vector<Param>{{"", DataCategory::Tensor, Precision::Float}},
    vector<Param>{{"", DataCategory::Tensor, Precision::Float}}));
```

### 6. 构建 + 验证

```bash
/build-exop-metal
```

检查输出:
- `[exop-metal] registered all ops` 包含 newop
- 构建无 Metal shader 编译错误

### 7. 测试

在 VM 侧编写 `.dx` 测试文件:
```
# testdata/native/arith/newop.dx
newtensor("./a") -> ./a   ← 创建输入
./a <- constant([3,3], 2.0)
newop(./a) -> ./b        ← 测试新算子
print(./b)
deltensor(./a)
deltensor(./b)
```

## 完成检查清单

- [ ] Metal shader 编译通过 (cmake 构建无错误)
- [ ] 所有声明 dtype 都有 kernel
- [ ] `/op/exop-metal/list` 包含新算子
- [ ] main.cpp dispatch 路径可达（无 unreachable code）
- [ ] 错误路径正确释放 shm + notify_done
- [ ] TfFactory 注册 (如适用)
