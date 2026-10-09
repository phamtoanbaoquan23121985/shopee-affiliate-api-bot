"""Independent overlapping fiber lanes; merge only at the terminal.
Each lane processes the same token sequence without cross-lane mixing.
"""
import torch
from torch import nn
import torch.nn.functional as F

class IndependentFiber(nn.Module):
    def __init__(self, width, rank, activation):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        self.activation=activation
        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        y=self.down(self.norm(x))
        if self.activation=="gelu": y=F.gelu(y)
        elif self.activation=="silu": y=F.silu(y)
        elif self.activation=="tanh": y=torch.tanh(y)
        else: raise ValueError("unknown fiber family")
        return x+self.up(y)

class TerminalFiberLayer(nn.Module):
    """Parallel, noncommunicating lanes with late fusion only."""
    def __init__(self,width=128,rank=8,depth=3,families=("gelu","silu","tanh")):
        super().__init__()
        if width<1 or rank<1 or depth<1 or not families:
            raise ValueError("invalid topology")
        self.lanes=nn.ModuleList([
            nn.Sequential(*(IndependentFiber(width,rank,f) for _ in range(depth)))
            for f in families])
        self.terminal_router=nn.Linear(width,len(families))
        self.terminal_norm=nn.LayerNorm(width)
        self.output_norm=nn.LayerNorm(width)
    def forward(self,x,available=None):
        if x.ndim!=3: raise ValueError("expected [batch,sequence,width]")
        lanes=torch.stack([lane(x) for lane in self.lanes],dim=-2)
        logits=self.terminal_router(self.terminal_norm(x))
        if available is not None:
            mask=torch.as_tensor(available,device=x.device,dtype=torch.bool)
            if mask.shape!=(len(self.lanes),) or not mask.any().item():
                raise ValueError("no active fiber")
            logits=logits.masked_fill(~mask,float("-inf"))
        weights=F.softmax(logits,dim=-1)
        fused=(lanes*weights.unsqueeze(-1)).sum(dim=-2)
        return self.output_norm(fused),{"terminal_weights":weights,"lanes":lanes}

class TerminalFiberLanguageHead(nn.Module):
    """Tokenwise architecture demonstration, not a causal LM."""
    def __init__(self,vocab_size,width=128,rank=8,depth=3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,width)
        self.fibers=TerminalFiberLayer(width,rank,depth)
        self.head=nn.Linear(width,vocab_size,bias=False)
        self.head.weight=self.embedding.weight
    def forward(self,token_ids,available=None):
        h=self.embedding(token_ids)
        fused,info=self.fibers(h,available)
        return self.head(fused),info
