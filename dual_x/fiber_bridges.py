"""Sparse low-rank cross-fiber microtubes at selected depth checkpoints."""
import torch
from torch import nn
import torch.nn.functional as F

class MicroTube(nn.Module):
    def __init__(self,width,rank):
        super().__init__()
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        self.gate=nn.Linear(width*2,1)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate.bias,-3.0)
    def forward(self,source,target):
        strength=torch.sigmoid(self.gate(torch.cat([source,target],dim=-1)))
        return strength*self.up(F.silu(self.down(source)))

class MicroTubeFiberNet(nn.Module):
    """Mostly independent lanes with explicit directed bridges at selected checkpoints.

    Edges are (depth, source_lane, destination_lane). Updates use snapshots,
    so all edges at one checkpoint are simultaneous, not order-dependent.
    """
    def __init__(self,width=64,rank=4,depth=4,lanes=4,edges=()):
        super().__init__()
        if min(width,rank,depth,lanes)<1: raise ValueError("invalid shape")
        self.width,self.depth,self.count=width,depth,lanes
        self.blocks=nn.ModuleList([nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width),nn.Linear(width,rank),nn.SiLU(),nn.Linear(rank,width))
            for _ in range(depth)]) for _ in range(lanes)])
        self.edges=tuple(tuple(edge) for edge in edges)
        if len(set(self.edges))!=len(self.edges): raise ValueError("duplicate edge")
        for d,s,t in self.edges:
            if not (0<=d<depth and 0<=s<lanes and 0<=t<lanes and s!=t):
                raise ValueError("invalid edge")
        self.bridges=nn.ModuleList([MicroTube(width,rank) for _ in self.edges])
        self.terminal=nn.Linear(width,lanes)
    def forward(self,x,enabled_edges=None):
        if x.ndim!=3 or x.shape[-1]!=self.width: raise ValueError("expected [batch,time,width]")
        states=[x for _ in range(self.count)]
        if enabled_edges is None: enabled_edges=[True]*len(self.edges)
        if len(enabled_edges)!=len(self.edges): raise ValueError("edge mask mismatch")
        for d in range(self.depth):
            states=[s+block[d](s) for s,block in zip(states,self.blocks)]
            snapshot=states
            additions=[torch.zeros_like(x) for _ in states]
            for i,(edge,tube) in enumerate(zip(self.edges,self.bridges)):
                checkpoint,source,target=edge
                if checkpoint==d and enabled_edges[i]:
                    additions[target]=additions[target]+tube(snapshot[source],snapshot[target])
            states=[s+delta for s,delta in zip(snapshot,additions)]
        stack=torch.stack(states,dim=-2)
        weights=torch.softmax(self.terminal(x),dim=-1)
        fused=(stack*weights.unsqueeze(-1)).sum(dim=-2)
        return fused,{"weights":weights,"bridge_count":sum(bool(v) for v in enabled_edges)}

def propose_edges(synergy, max_edges, threshold=0.0):
    """Use TRAIN-only validated, directed improvement matrix; diagonal forbidden."""
    n=len(synergy)
    if max_edges<0 or not all(len(row)==n for row in synergy): raise ValueError("invalid matrix")
    candidates=[]
    for s in range(n):
        for t in range(n):
            if s!=t and float(synergy[s][t])>threshold:
                candidates.append((float(synergy[s][t]),s,t))
    candidates.sort(reverse=True)
    return [(s,t) for _,s,t in candidates[:max_edges]]
