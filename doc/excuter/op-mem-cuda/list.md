## op-mem-cuda 支持算子列表 

本页面由 `excuter/op-mem-cuda 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| matmul | cublas | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1 @ T2 | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| rdivscalar | miaobyte | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) | T3=scalar/T1 | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) |
| div | miaobyte | div(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1/T2 | div(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| sub | miaobyte | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1-T2 | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |
| mulscalar | miaobyte | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1*scalar | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| vecset |  none  | vecset(vector<any> value)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> value)->(vector<any> name) |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| print | miaobyte | print(tensor<any> )->() | print(T1) | print(tensor<any> )->() |
| print | miaobyte | print(tensor<any> , var<string> )->() | print(T1) | print(tensor<any> , var<string> )->() |
| divscalar | miaobyte | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=scalar/T1 | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| constant | miaobyte | constant(tensor<any> t, var<any> value)->() | constant(T1) | constant(tensor<any> t, var<any> value)->() |
| arange | miaobyte | arange(tensor<any> t, var<any> start, var<any> step)->() | arange(T1,start,step) | arange(tensor<any> t, var<any> start, var<any> step)->() |
| subscalar | miaobyte | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1-scalar | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| uniform | miaobyte | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() | uniform(T1,low,high,seed) | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() |
| add | cublas | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| add | miaobyte | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| addscalar | miaobyte | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1+scalar | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| mul | miaobyte | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1*T2 | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
