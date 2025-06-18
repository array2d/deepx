from token_text import dir,config

############-------DEEPX-------################
from deepx.nn.modules import Embedding,Module
from deepx  import load,arange
from deepx.transformer.models.llama import rotate_half

input=load(dir+'input')
input.print()
r=rotate_half(input)
r.print()

