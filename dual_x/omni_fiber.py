"""General-purpose Nano Fiber foundation model prototype.

Shared causal backbone + specialized low-rank fibers + sparse microtube
cross-connections + task-conditioned terminal fusion. Random initialization:
pretraining and tool execution are not included.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class OmniConfig:
    vocab_size:int=32000
    width:int=192
    heads:int=6
    layers:int=6
    lanes:int=4
    rank:int=12
    max_context:int=512
    tasks:tuple=("language","reasoning","classification","retrieval","agent")

class CausalBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1=nn.LayerNorm(c.width)
        self.attn=nn.MultiheadAttention(c.width,c.heads,batch_first=True)
        self.n2=nn.LayerNorm(c.width)
        self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
    def forward(self,x):
        t=x.size(1)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        q=self.n1(x)
        a,_=self.attn(q,q,q,attn_mask=mask,need_weights=False)
        x=x+a
        return x+self.ff(self.n2(x))

class Fiber(nn.Module):
    def __init__(self,width,rank):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        return x+self.up(F.silu(self.down(self.norm(x))))

class OmniFiberLM(nn.Module):
    def __init__(self,c=OmniConfig()):
        super().__init__()
        if c.width%c.heads or min(c.layers,c.lanes,c.rank,c.max_context)<1:
            raise ValueError("invalid config")
        self.config=c
        self.embed=nn.Embedding(c.vocab_size,c.width)
        self.pos=nn.Embedding(c.max_context,c.width)
        self.backbone=nn.ModuleList([CausalBlock(c) for _ in range(c.layers)])
        self.fibers=nn.ModuleList([nn.ModuleList([Fiber(c.width,c.rank) for _ in range(c.layers)]) for _ in range(c.lanes)])
        # Ring-shaped directed microtubes at every second layer, not all-to-all.
        self.bridges=nn.ModuleDict({str(d):nn.ModuleList([Fiber(c.width,c.rank) for _ in range(c.lanes)]) for d in range(1,c.layers,2)})
        self.task_embed=nn.Embedding(len(c.tasks),c.width)
        self.router=nn.Linear(c.width,c.lanes)
        self.norm=nn.LayerNorm(c.width)
        self.lm_head=nn.Linear(c.width,c.vocab_size,bias=False)
        self.lm_head.weight=self.embed.weight
        self.classifier=nn.Linear(c.width,2)
        self.retrieval=nn.Linear(c.width,c.width)
        self.action=nn.Linear(c.width,8)
    def forward(self,tokens,task="language"):
        c=self.config
        if task not in c.tasks: raise ValueError("unknown task")
        if tokens.ndim!=2 or not 1<=tokens.size(1)<=c.max_context:
            raise ValueError("invalid token sequence")
        b,t=tokens.shape
        x=self.embed(tokens)+self.pos(torch.arange(t,device=tokens.device))[None,:,:]
        states=[x for _ in range(c.lanes)]
        for d,block in enumerate(self.backbone):
            x=block(x)
            states=[fiber[d](state+x) for fiber,state in zip(self.fibers,states)]
            if str(d) in self.bridges:
                snapshot=states
                states=[s+self.bridges[str(d)][(i-1)%c.lanes](snapshot[(i-1)%c.lanes])-snapshot[(i-1)%c.lanes]
                        for i,s in enumerate(snapshot)]
        context=x+self.task_embed.weight[c.tasks.index(task)]
        weights=F.softmax(self.router(context),dim=-1)
        fused=self.norm((torch.stack(states,dim=-2)*weights.unsqueeze(-1)).sum(-2))
        if task in ("language","reasoning"):
            output=self.lm_head(fused)
        elif task=="classification":
            output=self.classifier(fused[:, -1])
        elif task=="retrieval":
            output=F.normalize(self.retrieval(fused[:, -1]),dim=-1)
        else:
            output=self.action(fused[:, -1])
        return output,{"routing_weights":weights}
    def next_token_loss(self,tokens,task="language"):
        if tokens.size(1)<2: raise ValueError("need at least two tokens")
        logits,_=self(tokens[:,:-1],task)
        return F.cross_entropy(logits.reshape(-1,logits.size(-1)),tokens[:,1:].reshape(-1))
