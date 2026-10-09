"""Experimental input encoder and terminal decoder for Nano Fiber Graph.

Causal token mixing is handled by a masked convolution in the encoder.
The decoder combines lane states, then predicts next-token logits.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .fiber_bridges import MicroTubeFiberNet

class CausalStem(nn.Module):
    """Token embedding + strictly causal depthwise convolution."""
    def __init__(self,vocab_size,width=64,kernel_size=3):
        super().__init__()
        if vocab_size<2 or width<1 or kernel_size<1: raise ValueError("invalid stem dimensions")
        self.embed=nn.Embedding(vocab_size,width)
        self.depthwise=nn.Conv1d(width,width,kernel_size,groups=width,bias=False)
        self.gate=nn.Linear(width,width)
        self.norm=nn.LayerNorm(width)
        self.kernel_size=kernel_size
    def forward(self,tokens):
        if tokens.ndim!=2: raise ValueError("expected [batch,time]")
        h=self.embed(tokens)
        z=F.pad(h.transpose(1,2),(self.kernel_size-1,0))
        z=self.depthwise(z).transpose(1,2)
        return self.norm(h+torch.sigmoid(self.gate(h))*z)

class TerminalDecoder(nn.Module):
    """Residual gated output projection; tied embedding weights."""
    def __init__(self,width,vocab_size,embedding):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.refine=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.gate=nn.Linear(width,width)
        self.lm_head=nn.Linear(width,vocab_size,bias=False)
        self.lm_head.weight=embedding.weight
    def forward(self,h):
        z=self.norm(h)
        return self.lm_head(z+torch.sigmoid(self.gate(z))*self.refine(z))

class NanoFiberCodecLM(nn.Module):
    """End-to-end toy language model with input/output codecs and microtubes.

    Not a competitive long-context LLM: stem has a finite causal receptive field,
    and current fiber blocks are tokenwise.
    """
    def __init__(self,vocab_size,width=64,rank=4,depth=4,lanes=4,edges=()):
        super().__init__()
        self.encoder=CausalStem(vocab_size,width)
        self.network=MicroTubeFiberNet(width,rank,depth,lanes,edges)
        self.decoder=TerminalDecoder(width,vocab_size,self.encoder.embed)
    def forward(self,tokens,enabled_edges=None):
        h=self.encoder(tokens)
        h,meta=self.network(h,enabled_edges)
        return self.decoder(h),meta
    def next_token_loss(self,tokens):
        if tokens.ndim!=2 or tokens.size(1)<2:
            raise ValueError("at least two tokens required")
        logits,_=self(tokens[:,:-1])
        return F.cross_entropy(logits.reshape(-1,logits.size(-1)),tokens[:,1:].reshape(-1))
