# deepx 贡献指南

deepx框架的发展，主要包括五大类方向

+ front: 新增模型、module、python类函数等
+ 中间层：包括计算图优化器，插件系统(自动KVcache系统)，自动分布式化，栈tensor自动释放，自动Inplace化等操作
+ 新增或修改excuter
+ 增加或修改算子，进一步可以分为leaftensorfunc(不可分割的基础算子)，fusedtensorfunc（融合算子）
+ 文档丰富：
+ 运维自动化方向

大家可以选择一个方向

## 步骤

第一次提交
  1. Fork本仓库（github.com/array2d/deepx）的main分支，到你的github/yourname/deepx
  2. 本地clone github/yourname/deepx
  3. 提交并推送您的更改到你的github：`git commit -m 'Add some feature'`
  4. 创建一个Pull Request。

第N次提交 

  1. 保障你的本地和github/yourname/deepx中均已提pull request并得到merge
  2. 在github/yourname/deepx中sync fork【危险操作，会删除你新增的代码】，拉取（github.com/array2d/deepx） main分支的最新代码
  3. 本地clone github/yourname/deepx
  4. 提交并推送您的更改到你的github：`git commit -m 'Add some feature'`
  5. 创建一个Pull Request。