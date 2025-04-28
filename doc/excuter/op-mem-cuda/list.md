## op-mem-cuda 支持算子列表 

本页面由 `excuter/op-mem-cuda 生成，请勿手动修改 

### arg

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| vecset |  none  | [3  4  5]->shape | vecset(vector<any> value)->(vector<any> name) |
| argset |  none  | argvalue->argname | argset(var<any> value)->(var<any> name) |

### tensorlife

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| renametensor |  none  | rename(newname)->T1 | renametensor(var<string> new_name)->(tensor<any> t) |
| newtensor |  none  | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| deltensor |  none  | del->T1 | deltensor()->(tensor<any> t) |
| copytensor |  none  | T2.data = T1.data | copytensor(tensor<any> src)->(tensor<any> dst) |

### io

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| loadtensordata |  none  | loadtensordata(path)->tensor | loadtensordata(var<string> path)->(tensor<any> t) |
| save |  none  | save(T1,path) | save(tensor<any> t, var<string> path)->() |
| print | miaobyte | print(T1) | print(tensor<any> t)->() |
| print | miaobyte | print(T1) | print(tensor<any> t, var<string> format)->() |
| load |  none  | load(path) | load(var<string> path)->() |

### matmul

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| matmul | cublas | T3=T1 @ T2 | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) |

### init

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| normal | miaobyte | normal(mean,stddev,seed)->T1 | normal(var<any> mean, var<any> stddev, var<int32> seed)->(tensor<any> t) |
| uniform | miaobyte | uniform(low,high,seed)->T1 | uniform(var<any> low, var<any> high, var<int32> seed)->(tensor<any> t) |
| arange | miaobyte | arange(start,step)->T1 | arange(var<any> start, var<any> step)->(tensor<any> t) |
| constant | miaobyte | constant(value)->T1 | constant(var<any> value)->(tensor<any> t) |

### elementwise

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| switch | miaobyte | C=switch(tensors,cases) | switch(listtensor<any> tensors, tensor<int8> cases)->(tensor<any> result) |
| greaterscalar | miaobyte | mask=compare(T1, scalar) | greaterscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) |
| equalscalar | miaobyte | mask=compare(T1, scalar) | equalscalar(tensor<any> A, var<any> scalar, var<float64> epsilon)->(tensor<bool> mask) |
| min | miaobyte | T3=min(T1, T2) | min(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| maxscalar | miaobyte | T3=max(T1, scalar) | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| tan | miaobyte | T3=tan(T1) | tan(tensor<float64|float32> A)->(tensor<float64|float32> C) |
| sin | miaobyte | T3=sin(T1) | sin(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| dropout | miaobyte | dropout(p,seed)->A | dropout(var<float32> p, var<int32> seed)->(tensor<any> A) |
| divscalar | miaobyte | T3=scalar/T1 | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| log | miaobyte | T3=log(T1) | log(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| addscalar | miaobyte | T3=T1+scalar | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| greater | miaobyte | mask=compare(T1, T2) | greater(tensor<any> A, tensor<any> B)->(tensor<bool> mask) |
| lessscalar | miaobyte | mask=compare(T1, scalar) | lessscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) |
| cos | miaobyte | T3=cos(T1) | cos(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| minscalar | miaobyte | T3=min(T1, scalar) | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| rpowscalar | miaobyte | T3=pow(scalar, T1) | rpowscalar(var<float64|int32> scalar, tensor<float64|float32> A)->(tensor<float64|float32> C) |
| rdivscalar | miaobyte | T3=scalar/T1 | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) |
| less | miaobyte | mask=compare(T1, T2) | less(tensor<any> A, tensor<any> B)->(tensor<bool> mask) |
| powscalar | miaobyte | T3=pow(T1, scalar) | powscalar(tensor<float64|float32> A, var<float64|int32> scalar)->(tensor<float64|float32> C) |
| todtype |  none  | T3(dtypeA)->T1(dtypeB) | todtype(tensor<any> a)->(tensor<any> b) |
| add | cublas | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| add | miaobyte | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| sub | miaobyte | T3=T1-T2 | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| sqrt | miaobyte | T3=sqrt(T1) | sqrt(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| subscalar | miaobyte | T3=T1-scalar | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| exp | miaobyte | T3=exp(T1) | exp(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| mul | miaobyte | T3=T1*T2 | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| equal | miaobyte | mask=compare(T1, T2) | equal(tensor<any> A, tensor<any> B, var<float64> epsilon)->(tensor<bool> mask) |
| mulscalar | miaobyte | T3=T1*scalar | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| div | miaobyte | T3=T1/T2 | div(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| invert | miaobyte | T3=~T1 | invert(tensor<int64|int32|int16|int8> A)->(tensor<int64|int32|int16|int8> C) |
| max | miaobyte | T3=max(T1, T2) | max(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| pow | miaobyte | T3=pow(T1, T2) | pow(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) |

### reduce

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| prod | miaobyte | B = prod(A, axis=[1 2], keepdims=false) | prod(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| reducemax | miaobyte | B = reducemax(A, axis=[1 2], keepdims=false) | reducemax(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| sum | miaobyte | B = sum(A, axis=[1 2], keepdims=false) | sum(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| reducemin | miaobyte | B = reducemin(A, axis=[1 2], keepdims=false) | reducemin(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |

### changeshape

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| indexselect | miaobyte | T2 = T1.indexselect(index=[1,2], axis=1) | indexselect(tensor<any> A, tensor<int64|int32> indices, var<int32> axis)->(tensor<any> B) |
| broadcastTo | miaobyte | T2 = T1.broadcastTo(new_shape=[4,3,2]) | broadcastTo(tensor<any> A, vector<int32> new_shape)->(tensor<any> B) |
| concat | miaobyte | Tresult = concat([T1, T2...], axis=3) | concat(listtensor<any> tensors, var<int32> dim)->(tensor<any> result) |
| transpose | miaobyte | T2 = T1.transpose(dimorder=[1,0]) | transpose(tensor<any> A, vector<int32> dim_order)->(tensor<any> C) |
| reshape | miaobyte | T1.reshape(shape)->T2 | reshape(tensor<any> A, vector<int32> shape)->(tensor<any> B) |

