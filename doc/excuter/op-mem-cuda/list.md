## op-mem-cuda 支持算子列表 

本页面由 `excuter/op-mem-cuda 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| matmul | cublas | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1 @ T2 | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| add | cublas | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| add | miaobyte | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| uniform | miaobyte | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() | uniform(T1,low,high,seed) | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() |
| addscalar | miaobyte | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1+scalar | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| log | miaobyte | log(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=log(T1) | log(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| arange | miaobyte | arange(tensor<any> t, var<any> start, var<any> step)->() | arange(T1,start,step) | arange(tensor<any> t, var<any> start, var<any> step)->() |
| divscalar | miaobyte | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=scalar/T1 | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| sin | miaobyte | sin(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=sin(T1) | sin(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| tan | miaobyte | tan(tensor<float64|float32> A)->(tensor<float64|float32> C) | T3=tan(T1) | tan(tensor<float64|float32> A)->(tensor<float64|float32> C) |
| print | miaobyte | print(tensor<any> )->() | print(T1) | print(tensor<any> )->() |
| print | miaobyte | print(tensor<any> , var<string> )->() | print(T1) | print(tensor<any> , var<string> )->() |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| vecset |  none  | vecset(vector<any> value)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> value)->(vector<any> name) |
| subscalar | miaobyte | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1-scalar | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| sqrt | miaobyte | sqrt(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=sqrt(T1) | sqrt(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |
| sub | miaobyte | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1-T2 | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| mulscalar | miaobyte | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1*scalar | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| div | miaobyte | div(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1/T2 | div(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| constant | miaobyte | constant(tensor<any> t, var<any> value)->() | constant(T1) | constant(tensor<any> t, var<any> value)->() |
| powscalar | miaobyte | powscalar(tensor<float64|float32> A, var<float64|float32> scalar)->(tensor<float64|float32> C) | T3=pow(T1, scalar) | powscalar(tensor<float64|float32> A, var<float64|float32> scalar)->(tensor<float64|float32> C) |
| max | miaobyte | max(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=max(T1, T2) | max(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| pow | miaobyte | pow(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) | T3=pow(T1, T2) | pow(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) |
| maxscalar | miaobyte | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=max(T1, scalar) | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| mul | miaobyte | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1*T2 | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| exp | miaobyte | exp(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=exp(T1) | exp(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| rdivscalar | miaobyte | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) | T3=scalar/T1 | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) |
| minscalar | miaobyte | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=min(T1, scalar) | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| cos | miaobyte | cos(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=cos(T1) | cos(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| min | miaobyte | min(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=min(T1, T2) | min(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| compare | miaobyte | compare(tensor<any> A, tensor<any> B)->(tensor<int8> mask) | mask=compare(T1, T2) | compare(tensor<any> A, tensor<any> B)->(tensor<int8> mask) |
