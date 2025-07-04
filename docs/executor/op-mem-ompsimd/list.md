## op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd 生成，请勿手动修改 

### matmul

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| matmul | cblas | T3=T1 @ T2 | matmul(tensor<float64|float32>:A, tensor<float64|float32>:B)->(tensor<float64|float32>:C) |
| matmul | miaobyte | T3=T1 @ T2 | matmul(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |

### init

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| constant | miaobyte | constant(value)->T1 | constant(var<any>:value)->(tensor<any>:t) |
| arange | miaobyte | arange(start,step)->T1 | arange(var<any>:start, var<any>:step)->(tensor<any>:t) |
| uniform | miaobyte | uniform(low,high,seed)->T1 | uniform(var<any>:low, var<any>:high, var<int32>:seed)->(tensor<any>:t) |
| dropout | miaobyte | dropout(p,seed)->A | dropout(var<float32>:p, var<int32>:seed)->(tensor<any>:A) |
| normal | miaobyte | normal(mean,stddev,seed)->T1 | normal(var<any>:mean, var<any>:std, var<int32>:seed)->(tensor<any>:t) |

### io

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| load |  none  | mem.load(path) | load(var<string>:path)->() |
| print | miaobyte | print(T1) | print(tensor<any>:t)->() |
| print | miaobyte | print(T1) | print(tensor<any>:t, var<string>:format)->() |
| save |  none  | save(T1,path) | save(tensor<any>:t, var<string>:path)->() |
| loadtensordata |  none  | loadtensordata(path)->tensor.data | loadtensordata(var<string>:path)->(tensor<any>:t) |

### arg

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| argset |  none  | argvalue->argname | argset(var<any>:value)->(var<any>:name) |
| vecset |  none  | [3  4  5]->shape | vecset(vector<any>:value)->(vector<any>:name) |

### tensorlife

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| copytensor |  none  | T1.data->T2.data | copytensor(tensor<any>:src)->(tensor<any>:dst) |
| newtensor |  none  | T1 =Tensor(shape=[...]) | newtensor(vector<int32>:shape)->(tensor<any>:t) |
| newtensor |  none  | T1 =Tensor(shape=[...]) | newtensor(var<string>:shape)->(tensor<any>:t) |
| renametensor |  none  | rename(newname)->T1 | renametensor(var<string>:new_name)->(tensor<any>:t) |
| deltensor |  none  | del->T1 | deltensor()->(tensor<any>:t) |

### elementwise

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| pow | miaobyte | T3=T1^T2 | pow(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |
| max | miaobyte | T3=max(T1,T2) | max(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |
| mulscalar | miaobyte | T3=T1*scalar | mulscalar(tensor<any>:A, var<any>:b)->(tensor<any>:C) |
| equal | miaobyte | equal(T1,T2)->mask | equal(tensor<any>:A, tensor<any>:B, var<float32>:eposilon)->(tensor<bool>:mask) |
| mul | miaobyte | T3=T1*T2 | mul(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |
| exp | miaobyte | T3=exp(T1) | exp(tensor<any>:A)->(tensor<any>:C) |
| subscalar | miaobyte | T3=T1-scalar | subscalar(tensor<any>:a, var<any>:scalar)->(tensor<any>:c) |
| sqrt | miaobyte | T3=sqrt(T1) | sqrt(tensor<any>:A)->(tensor<any>:C) |
| sub | miaobyte | T3=T1-T2 | sub(tensor<any>:a, tensor<any>:b)->(tensor<any>:c) |
| add | cblas | T3=T1+T2 | add(tensor<float64|float32>:a, tensor<float64|float32>:b)->(tensor<float64|float32>:c) |
| add | miaobyte | T3=T1+T2 | add(tensor<any>:a, tensor<any>:b)->(tensor<any>:c) |
| todtype |  none  | T3(dtypeA)->T1(dtypeB) | todtype(tensor<any>:A)->(tensor<any>:C) |
| invert | miaobyte | T3=~T1 | invert(tensor<int64|int32|int16|int8|bool>:A)->(tensor<int64|int32|int16|int8|bool>:C) |
| rdivscalar | miaobyte | T3=scalar/T1 | rdivscalar(var<any>:scalar, tensor<any>:A)->(tensor<any>:C) |
| rpowscalar | miaobyte | T3=scalar^T1 | rpowscalar(var<float32>:scalar, tensor<any>:A)->(tensor<any>:C) |
| cos | miaobyte | T3=cos(T1) | cos(tensor<any>:A)->(tensor<any>:C) |
| greater | miaobyte | mask=greater(T1,T2) | greater(tensor<any>:A, tensor<any>:B)->(tensor<bool>:mask) |
| addscalar | miaobyte | T3=T1+scalar | addscalar(tensor<any>:a, var<any>:scalar)->(tensor<any>:c) |
| div | miaobyte | T3=T1/T2 | div(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |
| divscalar | miaobyte | T3=T1/scalar | divscalar(tensor<any>:A, var<any>:scalar)->(tensor<any>:C) |
| lessscalar | miaobyte | mask=less(T1,scalar) | lessscalar(tensor<any>:A, var<any>:scalar)->(tensor<bool>:mask) |
| less | miaobyte | mask=less(T1,T2) | less(tensor<any>:A, tensor<any>:B)->(tensor<bool>:mask) |
| notequalscalar | miaobyte | mask=notequal(T1,scalar) | notequalscalar(tensor<any>:A, var<any>:scalar, var<float32>:epsilon)->(tensor<bool>:mask) |
| rsubscalar | miaobyte | T3=scalar-T1 | rsubscalar(var<any>:scalar, tensor<any>:a)->(tensor<any>:c) |
| sin | miaobyte | T3=sin(T1) | sin(tensor<any>:A)->(tensor<any>:C) |
| minscalar | miaobyte | T3=min(T1,scalar) | minscalar(tensor<any>:A, var<any>:scalar)->(tensor<any>:C) |
| tan | miaobyte | T3=tan(T1) | tan(tensor<any>:A)->(tensor<any>:C) |
| maxscalar | miaobyte | T3=max(T1,scalar) | maxscalar(tensor<any>:A, var<any>:scalar)->(tensor<any>:C) |
| min | miaobyte | T3=min(T1,T2) | min(tensor<any>:A, tensor<any>:B)->(tensor<any>:C) |
| log | miaobyte | T3=log(T1) | log(tensor<any>:A)->(tensor<any>:C) |
| equalscalar | miaobyte | mask=equal(T1,scalar) | equalscalar(tensor<any>:A, var<any>:scalar, var<float32>:eposilon)->(tensor<bool>:mask) |
| notequal | miaobyte | notequal(T1,T2)->mask | notequal(tensor<any>:A, tensor<any>:B, var<float32>:epsilon)->(tensor<bool>:mask) |
| greaterscalar | miaobyte | mask=greater(T1,scalar) | greaterscalar(tensor<any>:A, var<any>:scalar)->(tensor<bool>:mask) |
| switch | miaobyte | C=switch([tensors],case) | switch(listtensor<any>:tensors, tensor<int32|bool>:cases)->(tensor<any>:C) |
| powscalar | miaobyte | T3=T1^scalar | powscalar(tensor<any>:A, var<any>:scalar)->(tensor<any>:C) |

### changeshape

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| reshape | miaobyte | T1.reshape(shape)->T2 | reshape(tensor<any>:A, vector<int32>:shape)->(tensor<any>:B) |
| transpose | miaobyte | T1.transpose(dimorder=[1,0])->T2 | transpose(tensor<any>:A, vector<int32>:dim_order)->(tensor<any>:C) |
| concat | miaobyte | Tresult = concat([T1, T2...], axis=3) | concat(listtensor<any>:tensors, var<int32>:dim)->(tensor<any>:result) |
| broadcastTo | miaobyte | T2 = T1.broadcastTo(new_shape=[4,3,2]) | broadcastTo(tensor<any>:A, vector<int32>:new_shape)->(tensor<any>:B) |
| indexselect | miaobyte | T2 = T1.indexselect(index=T3, axis=3) | indexselect(tensor<any>:A, tensor<int64|int32>:index, var<int32>:axis)->(tensor<any>:B) |
| repeat | miaobyte | T1.repeat(repeats=[4,3,2])->T2 | repeat(tensor<any>:A, vector<int32>:repeats)->(tensor<any>:B) |

### reduce

| Operation | Author |  Math Formula | IR Instruction |
|-----------|--------|--------------|----------------|
| reducemin | miaobyte | B = reducemin(A, axis=[1 2], keepdims=false) | reducemin(tensor<any>:A, vector<int32>:axis, var<bool>:keepdims)->(tensor<any>:B) |
| sum | miaobyte | B = sum(A, axis=[1 2], keepdims=false) | sum(tensor<any>:A, vector<int32>:axis, var<bool>:keepdims)->(tensor<any>:B) |
| reducemax | miaobyte | B = reducemax(A, axis=[1 2], keepdims=false) | reducemax(tensor<any>:A, vector<int32>:axis, var<bool>:keepdims)->(tensor<any>:B) |
| prod | miaobyte | B = prod(A, axis=[1 2], keepdims=false) | prod(tensor<any>:A, vector<int32>:axis, var<bool>:keepdims)->(tensor<any>:B) |

