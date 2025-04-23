## op-mem-cuda 支持算子列表 

本页面由 `excuter/op-mem-cuda 生成，请勿手动修改 

### arg

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| vecset |  none  | vecset(vector<any> value)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> value)->(vector<any> name) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |

### tensorlife

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| renametensor |  none  | renametensor(tensor<any> t, var<string> new_name)->() | rename T1 to T2 | renametensor(tensor<any> t, var<string> new_name)->() |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| deltensor |  none  | deltensor(tensor<any> t)->() | del T1 | deltensor(tensor<any> t)->() |
| copytensor |  none  | copytensor(tensor<any> src, tensor<any> dst)->() | T2.data = T1.data | copytensor(tensor<any> src, tensor<any> dst)->() |

### io

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| save |  none  | save(tensor<any> t, var<string> path)->() | save(T1,path) | save(tensor<any> t, var<string> path)->() |
| print | miaobyte | print(tensor<any> t)->() | print(T1) | print(tensor<any> t)->() |
| print | miaobyte | print(tensor<any> t, var<string> format)->() | print(T1) | print(tensor<any> t, var<string> format)->() |
| load |  none  | load(var<string> path)->() | load(path) | load(var<string> path)->() |

### init

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| normal | miaobyte | normal(tensor<any> t, var<any> mean, var<any> stddev, var<int32> seed)->() | normal(T1,mean,stddev,seed) | normal(tensor<any> t, var<any> mean, var<any> stddev, var<int32> seed)->() |
| uniform | miaobyte | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() | uniform(T1,low,high,seed) | uniform(tensor<any> t, var<any> low, var<any> high, var<int32> seed)->() |
| arange | miaobyte | arange(tensor<any> t, var<any> start, var<any> step)->() | arange(T1,start,step) | arange(tensor<any> t, var<any> start, var<any> step)->() |
| constant | miaobyte | constant(tensor<any> t, var<any> value)->() | constant(T1) | constant(tensor<any> t, var<any> value)->() |

### elementwise

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| switch | miaobyte | switch(listtensor<any> tensors, tensor<int8> cases)->(tensor<any> result) | C=switch(tensors,cases) | switch(listtensor<any> tensors, tensor<int8> cases)->(tensor<any> result) |
| greaterscalar | miaobyte | greaterscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) | mask=compare(T1, scalar) | greaterscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) |
| equalscalar | miaobyte | equalscalar(tensor<any> A, var<any> scalar, var<float64> epsilon)->(tensor<bool> mask) | mask=compare(T1, scalar) | equalscalar(tensor<any> A, var<any> scalar, var<float64> epsilon)->(tensor<bool> mask) |
| min | miaobyte | min(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=min(T1, T2) | min(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| maxscalar | miaobyte | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=max(T1, scalar) | maxscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| tan | miaobyte | tan(tensor<float64|float32> A)->(tensor<float64|float32> C) | T3=tan(T1) | tan(tensor<float64|float32> A)->(tensor<float64|float32> C) |
| sin | miaobyte | sin(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=sin(T1) | sin(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| divscalar | miaobyte | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=scalar/T1 | divscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| log | miaobyte | log(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=log(T1) | log(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| addscalar | miaobyte | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1+scalar | addscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| greater | miaobyte | greater(tensor<any> A, tensor<any> B)->(tensor<bool> mask) | mask=compare(T1, T2) | greater(tensor<any> A, tensor<any> B)->(tensor<bool> mask) |
| lessscalar | miaobyte | lessscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) | mask=compare(T1, scalar) | lessscalar(tensor<any> A, var<any> scalar)->(tensor<bool> mask) |
| cos | miaobyte | cos(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=cos(T1) | cos(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| less | miaobyte | less(tensor<any> A, tensor<any> B)->(tensor<bool> mask) | mask=compare(T1, T2) | less(tensor<any> A, tensor<any> B)->(tensor<bool> mask) |
| powscalar | miaobyte | powscalar(tensor<float64|float32> A, var<float64|int32> scalar)->(tensor<float64|float32> C) | T3=pow(T1, scalar) | powscalar(tensor<float64|float32> A, var<float64|int32> scalar)->(tensor<float64|float32> C) |
| minscalar | miaobyte | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) | T3=min(T1, scalar) | minscalar(tensor<any> A, var<any> scalar)->(tensor<any> C) |
| rdivscalar | miaobyte | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) | T3=scalar/T1 | rdivscalar(var<any> scalar, tensor<any> A)->(tensor<any> C) |
| rpowscalar | miaobyte | rpowscalar(var<float64|int32> scalar, tensor<float64|float32> A)->(tensor<float64|float32> C) | T3=pow(scalar, T1) | rpowscalar(var<float64|int32> scalar, tensor<float64|float32> A)->(tensor<float64|float32> C) |
| add | cublas | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| add | miaobyte | add(tensor<any> a, tensor<any> b)->(tensor<any> c) | T3=T1+T2 | add(tensor<any> a, tensor<any> b)->(tensor<any> c) |
| sub | miaobyte | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1-T2 | sub(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| sqrt | miaobyte | sqrt(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=sqrt(T1) | sqrt(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |
| subscalar | miaobyte | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1-scalar | subscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| equal | miaobyte | equal(tensor<any> A, tensor<any> B, var<float64> epsilon)->(tensor<bool> mask) | mask=compare(T1, T2) | equal(tensor<any> A, tensor<any> B, var<float64> epsilon)->(tensor<bool> mask) |
| mulscalar | miaobyte | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) | T3=T1*scalar | mulscalar(tensor<any> A, var<any> b)->(tensor<any> C) |
| div | miaobyte | div(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1/T2 | div(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| invert | miaobyte | invert(tensor<int64|int32|int16|int8> A)->(tensor<int64|int32|int16|int8> C) | T3=~T1 | invert(tensor<int64|int32|int16|int8> A)->(tensor<int64|int32|int16|int8> C) |
| max | miaobyte | max(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=max(T1, T2) | max(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| pow | miaobyte | pow(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) | T3=pow(T1, T2) | pow(tensor<float64|float32> A, tensor<float64|float32> B)->(tensor<float64|float32> C) |
| mul | miaobyte | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1*T2 | mul(tensor<any> A, tensor<any> B)->(tensor<any> C) |
| exp | miaobyte | exp(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) | T3=exp(T1) | exp(tensor<float64|float32|float16|bfloat16> A)->(tensor<float64|float32|float16|bfloat16> C) |

### matmul

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| matmul | cublas | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) | T3=T1 @ T2 | matmul(tensor<any> A, tensor<any> B)->(tensor<any> C) |

### reduce

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| prod | miaobyte | prod(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) | B = prod(A, axis=[1 2], keepdims=false) | prod(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| reducemax | miaobyte | reducemax(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) | B = reducemax(A, axis=[1 2], keepdims=false) | reducemax(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| sum | miaobyte | sum(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) | B = sum(A, axis=[1 2], keepdims=false) | sum(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |
| reducemin | miaobyte | reducemin(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) | B = reducemin(A, axis=[1 2], keepdims=false) | reducemin(tensor<any> A, vector<int32> dims, var<bool> keepdims)->(tensor<any> B) |

### changeshape

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| indexselect | miaobyte | indexselect(tensor<any> A, tensor<int64|int32> indices, var<int32> axis)->(tensor<any> B) | T2 = T1.indexselect(index=[1,2], axis=1) | indexselect(tensor<any> A, tensor<int64|int32> indices, var<int32> axis)->(tensor<any> B) |
| broadcastTo | miaobyte | broadcastTo(tensor<any> A, vector<int32> new_shape)->(tensor<any> B) | T2 = T1.broadcastTo(new_shape=[4,3,2]) | broadcastTo(tensor<any> A, vector<int32> new_shape)->(tensor<any> B) |
| concat | miaobyte | concat(listtensor<any> tensors, var<int32> dim)->(tensor<any> result) | Tresult = concat([T1, T2...], axis=3) | concat(listtensor<any> tensors, var<int32> dim)->(tensor<any> result) |
| transpose | miaobyte | transpose(tensor<any> A, vector<int32> dim_order)->(tensor<any> C) | T2 = T1.transpose(dimorder=[1,0]) | transpose(tensor<any> A, vector<int32> dim_order)->(tensor<any> C) |
| reshape | miaobyte | reshape(tensor<any> A, vector<int32> shape)->(tensor<any> B) | T1.reshape(shape)->T2 | reshape(tensor<any> A, vector<int32> shape)->(tensor<any> B) |

