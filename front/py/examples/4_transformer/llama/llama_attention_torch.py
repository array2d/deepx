from token_text import torch_input
print()
############-------TORCH-------################
from  transformers.models.llama.modeling_llama import rotate_half

print(torch_input)
r=rotate_half(torch_input)
print(r)
