## op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| concat |  none  | concat()->() | Tresult = concat([T1, T2...], axis=3) | concat()->() |
| matmul | cblas | matmul(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) | T3=T1 @ T2 | matmul(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) |
| matmul | miaobyte | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1 @ T2 | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| compare | miaobyte | compare(tensor<any> A, tensor<any> B)->(tensor<int8> mask) | mask=compare(T1,T2) | compare(tensor<any> A, tensor<any> B)->(tensor<int8> mask) |
| min | miaobyte | min(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=min(T1,T2) | min(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| minscalar | miaobyte | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=min(T1,scalar) | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| exp | miaobyte | exp(tensor<any> A)->(tensor<any> C) | T3=exp(T1) | exp(tensor<any> A)->(tensor<any> C) |
| maxscalar | miaobyte | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=max(T1,scalar) | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| pow | miaobyte | pow(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1^T2 | pow(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| powscalar | miaobyte | powscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=T1^scalar | powscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| rdivscalar | miaobyte | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) | T3=scalar/T1 | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) |
| div | miaobyte | div(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1/T2 | div(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| sub | miaobyte | sub(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1-T2 | sub(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |
| mulscalar | miaobyte | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1*scalar | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| sqrt | miaobyte | sqrt(tensor<any> A)->(tensor<any> C) | T3=sqrt(T1) | sqrt(tensor<any> A)->(tensor<any> C) |
| vecset |  none  | vecset(vector<any> value)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> value)->(vector<any> name) |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 =Tensor(shape=[...]) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 =Tensor(shape=[...]) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| print | miaobyte | print(tensor<any> )->() | print(T1) | print(tensor<any> )->() |
| print | miaobyte | print(tensor<any> , var<string> )->() | print(T1) | print(tensor<any> , var<string> )->() |
| max | miaobyte | max(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=max(T1,T2) | max(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| divscalar | miaobyte | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=T1/scalar | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| constant | miaobyte | constant(tensor<any> t, var<any> value)->() | constant(T1,value) | constant(tensor<any> t, var<any> value)->() |
| arange | miaobyte | arange(tensor<any> t, var<any> start, var<any> step)->() | arange(T1,start,step) | arange(tensor<any> t, var<any> start, var<any> step)->() |
| subscalar | miaobyte | subscalar(tensor<any> a, var<any> scalar)->(tensor<any> c) | T3=T1-scalar | subscalar(tensor<any> a, var<any> scalar)->(tensor<any> c) |
| log | miaobyte | log(tensor<any> A)->(tensor<any> C) | T3=log(T1) | log(tensor<any> A)->(tensor<any> C) |
| uniform | miaobyte | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() | uniform(T1,low,high,seed) | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() |
| add | cblas | add(tensor<float64|float32> a, tensor<float64|float32> b)->(tensor<float64|float32> c) | T3=T1+T2 | add(tensor<float64|float32> a, tensor<float64|float32> b)->(tensor<float64|float32> c) |
| add | miaobyte | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| addscalar | miaobyte | addscalar(tensor<any> a, var<any> scalar)->(tensor<any> c) | T3=T1+scalar | addscalar(tensor<any> a, var<any> scalar)->(tensor<any> c) |
| mul | miaobyte | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1*T2 | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
