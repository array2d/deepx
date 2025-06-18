from token_text import dir

############-------DEEPX-------################
from deepx  import load
from deepx.transformer.models.llama import rotate_half

input=load(dir+'input')
input.print()
r=rotate_half(input)
r.print()

