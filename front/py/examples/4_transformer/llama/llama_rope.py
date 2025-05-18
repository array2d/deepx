from llama_rope_torch import dir,config

############-------DEEPX-------################
from deepx.nn.modules import Embedding,Module
from deepx  import load,arange
from deepx.transformer.models.llama import LlamaRotaryEmbedding

input=load(dir+'input')

embed_tokens_weight=load(dir+'weight')

class NetDeepx(Module):
    def __init__(self,configdict:dict):
        super().__init__()
        self.embed_tokens = Embedding(configdict["vocab_size"], configdict["hidden_size"],weight=embed_tokens_weight)
        self.rotary_emb = LlamaRotaryEmbedding(config=configdict)
        print("rotary_emb.inv_freq")
        self.rotary_emb.inv_freq.print()
    def forward(self,x):
        inputs_embeds = self.embed_tokens(x)
        hidden_states = inputs_embeds
        position_ids = arange(start=0,end=hidden_states.shape[1]).unsqueeze(0)
        return self.rotary_emb(hidden_states, position_ids)

if __name__ == "__main__":
    net = NetDeepx(configdict=config.to_dict())
    out=net.forward(input)
    out[0].print()
    out[1].print()


