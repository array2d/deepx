### 2025-01-9
deepx第三次重构
目标：性能与特性并重


### 2025-01-17
尝试omp+highway的simd融合

### 2025-01-20

layer.Node需要仔细设计forward和backward的接口

+ 输入输出用string作为key，从tensormanager中获取tensor
+ parallel结构

### 2025-01-21
h5模型文件，转deepx格式

### 2025-02-06

op完全重构

+ 输入输出用string作为key，从tensormanager中获取tensor

+ 对算子的精度进行了特化


### 2025-02-07

+  关于simd对齐的3段式对齐
 ```
    头部未对齐：通过标量运算处理直到对齐边界

      const size_t adjust = (alignment - misalign) / sizeof(T);
   for (; j < adjust...)


    主体对齐部分：使用对齐加载/存储指令


      Load(tag, a_start + j);  // 对齐加载
   Store(...);  // 对齐存储


    尾部剩余元素：处理最后不足一个向量宽度的元素

    
      for (; j < len; ++j)
 ```

### 2025-02-19

+ 增加、优化了部分tensorfunc的性能
+ 验证了一些列op的正确性

