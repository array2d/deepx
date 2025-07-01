from deepx.utils import Config
from token_text import dir,config

############-------DEEPX-------################
from deepx.nn.modules import Embedding,Module
from deepx  import load,arange
from deepx.nn.modules.transformer  import LlamaRotaryEmbedding

input=load(dir+'input')

embed_tokens_weight=load(dir+'weight')

class NetDeepx(Module):
    def __init__(self,config:Config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size,weight=embed_tokens_weight)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
    def forward(self,x):
        inputs_embeds = self.embed_tokens(x)
        hidden_states = inputs_embeds
        position_ids = arange(start=0,end=hidden_states.shape[1]).unsqueeze(0)
        return self.rotary_emb(hidden_states, position_ids)

if __name__ == "__main__":
    net = NetDeepx(config=config)
    out=net.forward(input)
    out[0].print()
    out[1].print()


