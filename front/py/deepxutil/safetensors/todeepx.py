from safetensors import safe_open
import numpy as np
import os
import json
import argparse
import shutil
import glob
import re

class TensorInfo:
    def __init__(self, dtype, ndim, shape, size, strides=None):
        self.dtype = dtype  # 数据精度类型，如"float32"
        self.ndim = ndim  # 维度数
        self.shape = shape  # 形状元组
        self.size = size  # 总元素数量
        self.strides = strides  # 步长数组（可选）


class Tensor:
    def __init__(self, data, tensorinfo: TensorInfo):
        assert isinstance(tensorinfo, TensorInfo),"tensorinfo必须是TensorInfo实例"
        self.data = data
        self.tensorinfo = tensorinfo

class SafeTensorExporter:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.config = self._load_config()


    def _load_config(self):
        """加载模型配置"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _find_model_files(self)->list:
        """查找所有分片模型文件"""
        single_file = os.path.join(self.model_dir, "model.safetensors")
        shard_files = glob.glob(os.path.join(self.model_dir, "model-*-of-*.safetensors"))
        
        # 使用正则表达式提取分片编号
        pattern = re.compile(r"model-(\d+)-of-(\d+)\.safetensors")
        filtered_shards = []
        for f in shard_files:
            match = pattern.search(os.path.basename(f))
            if match:
                filtered_shards.append( (int(match.group(1)), f) )
        
        if os.path.exists(single_file):
            return [single_file]
        elif filtered_shards:
            # 按分片编号排序后返回路径
            filtered_shards.sort(key=lambda x: x[0])
            return [f[1] for f in filtered_shards]
        raise FileNotFoundError(f"No model files found in {self.model_dir}")

    def export(self):
        """导出safetensor模型到指定目录"""
        model_files = self._find_model_files()
        
        from deepxutil.numpy import save_numpy

        for model_path in model_files:
            with safe_open(model_path, framework="numpy") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    path= os.path.join(self.output_dir, key)
                    save_numpy(t,path)

        self.mvothers()
        
    def mvothers(self):
        """复制tokenizer、config.json等相关文件到输出目录"""
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json"
        ]
        
        for filename in required_files:
            src = os.path.join(self.model_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.output_dir, filename))


if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser(description='Safetensor模型转换工具')
    parser.add_argument('model', type=str, 
                        help='输入目录路径，包含model.safetensors和config.json')
    parser.add_argument('--output','-o', type=str, required=True,
                        help='输出目录路径，转换后的DeepX格式数据将保存于此')

    args = parser.parse_args()

    exporter = SafeTensorExporter(
        model_dir=args.model,
        output_dir=args.output
    )
    try:
        exporter.export()
        print(f"转换成功！输出目录：{args.output_dir}")
    except Exception as e:
        print(f"转换失败：{str(e)}")
        exit(1)