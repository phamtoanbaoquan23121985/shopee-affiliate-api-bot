"""Heterogeneous low-rank expert fibers, PyTorch prototype.
No pretrained weights are magically compressed: distillation is a separate training step.
"""
import torch
from torch import nn
import torch.nn.functional as F

class LowRankExpert(nn.Module):
    def __init__(self, width, rank, hidden):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.gate = nn.Linear(width, rank)
        self.norm = nn.LayerNorm(width)
        self.hidden = hidden
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        h = self.down(self.norm(x))
        if self.hidden == "gelu":
            h = F.gelu(h)
        elif self.hidden == "silu":
            h = F.silu(h)
        elif self.hidden == "tanh":
            h = torch.tanh(h)
        else:
            raise ValueError("unknown expert family")
        return self.up(h * torch.sigmoid(self.gate(x)))

class FiberMoELayer(nn.Module):
    def __init__(self, width=128, rank=8, top_k=2, families=("gelu","silu","tanh")):
        super().__init__()
        if width <= 0 or rank <= 0 or not 1 <= top_k <= len(families):
            raise ValueError("invalid architecture")
        self.experts = nn.ModuleList([LowRankExpert(width,rank,f) for f in families])
        self.router = nn.Linear(width,len(families))
        self.top_k = top_k
        self.norm = nn.LayerNorm(width)
    def forward(self,x,available=None):
        if x.ndim != 3:
            raise ValueError("expected [batch, sequence, width]")
        logits = self.router(self.norm(x))
        if available is not None:
            mask = torch.as_tensor(available,dtype=torch.bool,device=x.device)
            if mask.shape != (len(self.experts),) or mask.sum().item() < self.top_k:
                raise ValueError("insufficient available fibers")
            logits = logits.masked_fill(~mask,float("-inf"))
        values, indices = torch.topk(logits,self.top_k,dim=-1)
        weights = F.softmax(values,dim=-1)
        flat = x.reshape(-1,x.shape[-1])
        idx = indices.reshape(-1,self.top_k)
        probs = weights.reshape(-1,self.top_k)
        delta = torch.zeros_like(flat)
        for j,expert in enumerate(self.experts):
            positions,slots = (idx==j).nonzero(as_tuple=True)
            if positions.numel():
                update = expert(flat.index_select(0,positions))
                delta.index_add_(0,positions,update*probs[positions,slots].unsqueeze(-1))
        return x+delta.reshape_as(x), {"indices":indices,"weights":weights}

class NanoFiberAgent(nn.Module):
    def __init__(self,vocab_size,width=128,layers=2,rank=8,top_k=2):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,width)
        self.blocks=nn.ModuleList([FiberMoELayer(width,rank,top_k) for _ in range(layers)])
        self.output=nn.Linear(width,vocab_size,bias=False)
        self.output.weight=self.embedding.weight
    def forward(self,token_ids,available=None):
        h=self.embedding(token_ids)
        routes=[]
        for block in self.blocks:
            h,route=block(h,available)
            routes.append(route)
        return self.output(h),routes
