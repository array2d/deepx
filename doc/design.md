# deepx默认原则

## 一.DeepxIR

### 1.deepIR结构
```
deepIR{
    Meta{
        int id
        string author
    } meta
    string name 
    []Param args
    []Param returns
}
```

excuter执行deepxIR的规则

+ excuter执行deepxIR时，不得修改args中的tensor
+ 但deepIR不限制args和returns中的Param同名，这样可以实现类似inplace的操作


## front/python规则

### 1.命名规则
+ inplace操作的函数，其名为_后缀, 返回值为空
+ 非inplace操作的函数，其名无_后缀
