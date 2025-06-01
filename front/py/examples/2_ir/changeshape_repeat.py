import torch

# 正确：repeats为一维张量
x = torch.tensor([[1, 2], [3, 4]])
repeats = torch.tensor([1, 2])  # 一维张量
torch.repeat_interleave(x, repeats, dim=0)
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [3, 4]])

# 错误：repeats为二维张量
repeats_2d = torch.tensor([[1, 2], [3, 4]])  # 二维张量
try:
    torch.repeat_interleave(x, repeats_2d, dim=0)
except RuntimeError as e:
    print(f"错误: {e}")
# 输出: