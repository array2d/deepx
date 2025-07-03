from safetensor_deepx import SafeTensorLoader, SafeTensorGraphBuilder
import os

def main():
    model_dir = "/home/lipeng/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 加载tensor
    loader = SafeTensorLoader(model_dir)
    tensors, metadata = loader.load()
    
    print("\nModel Configuration:")
    for key, value in metadata.get("model_config", {}).items():
        print(f"{key}: {value}")
    
    print("\nTensor Statistics:")
    total_params = 0
    for name, tensor in tensors.items():
        shape = tensor.shape
        num_params = tensor.data.size
        total_params += num_params
        print(f"{name}: shape={shape}, params={num_params:,}")
    
    print(f"\nTotal Parameters: {total_params:,}")
        
    # 构建计算图
    builder = SafeTensorGraphBuilder(model_dir)
    graph, _, _ = builder.build_graph()
    
    # 导出计算图可视化
    output_dir = "model_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    dot = graph.to_dot()
    dot.render(os.path.join(output_dir, "model_graph"), format="png", cleanup=True)
    
    print(f"\n计算图已保存到 {output_dir}/model_graph.png")

if __name__ == "__main__":
    main() 