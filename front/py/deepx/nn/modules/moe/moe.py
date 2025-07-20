from typing import Callable, List, Optional, Type, Union
from deepx.nn.modules import Module, Linear
from deepx.nn.modules.mlp.mlp import MLP, StandardMLP
from deepx import Tensor, nonzero
from deepx.utils import Config
import math

class MixtureOfExperts(Module):
    """
    MoE (Mixture of Experts) 层实现
    
    Args:
        config (dict): 配置参数
        expert_class (Type[Module]): 专家类型，默认使用 MLP
        num_experts (int): 专家数量
        top_k (int): 每个token激活的专家数量
        capacity_factor (float): 容量因子，控制每个专家的最大token数
    """
    def __init__(
        self, 
        config: Config,
        expert_class: Type[Module] = MLP,
        num_experts: int = 8, 
        top_k: int = 2,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
 
        # 计算每个专家的容量上限
        self.capacity_factor = capacity_factor
        
        # 创建专家网络
        self.experts = [expert_class(config) for _ in range(num_experts)]
        for i, expert in enumerate(self.experts):
            self.register_module(f"expert_{i}", expert)
        
        # 创建路由门
        self.gate = Linear(self.hidden_size, num_experts, bias=False)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        MoE 前向计算
        
        Args:
            hidden_states (Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor: 输出张量，形状与输入相同
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # 将输入展平为 [batch_size*seq_len, hidden_size]
        inputs = hidden_states.reshape(-1, hidden_size)
        
        # 计算路由门得分 [batch_size*seq_len, num_experts]
        router_logits = self.gate(inputs)
 
        routing_weights, selected_experts = self._compute_topk_routing(router_logits)
        # 路由计算
        outputs = self._dispatch_and_combine(inputs, routing_weights, selected_experts)

        # 恢复原始形状
        return outputs.reshape(batch_size, seq_len, hidden_size)
    
    def _compute_topk_routing(self, router_logits: Tensor):
        """计算top-k路由权重和专家索引"""
        # 使用softmax获取概率分布
        router_probs = router_logits.softmax(dim=-1)
        
        # 获取top-k及其索引
        # 注意: 实际实现会根据深度学习框架API调整
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        
        # 重新归一化权重
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices
 
    def _dispatch_and_combine(self, inputs: Tensor, routing_weights: Tensor, selected_experts: Tensor):
        """
        将输入分发给专家并结合结果
        
        这里使用的是简化版的实现，实际的MoE库有更复杂的负载均衡和高效实现
        """
        batch_tokens = inputs.shape[0]  # batch_size * seq_len
        output = 0
        
        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            # 找到路由到当前专家的所有token位置
            # 这里简化处理，实际实现会使用更高效的操作
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                # 如果没有token路由到该专家，则跳过
                continue
            
            # 提取需要送入该专家的token
            expert_inputs = inputs[expert_mask]
            if expert_inputs.shape[0] == 0:
                continue
                
            # 计算专家输出
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # 计算路由权重
            # 找出每个选中token对应于当前专家的权重位置
            weight_mask = (selected_experts == expert_idx)
            
            # 提取对应的权重
            expert_weights = weight_mask.float() * routing_weights
            expert_weights = expert_weights.sum(dim=-1)[expert_mask].unsqueeze(-1)
            
            # 加权组合
            output_indices = nonzero(expert_mask).squeeze(-1)
            for i, idx in enumerate(output_indices):
                output[idx] += expert_output[i] * expert_weights[i]
        
        return output
