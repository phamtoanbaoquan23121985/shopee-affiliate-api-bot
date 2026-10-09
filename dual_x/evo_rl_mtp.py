"""Experimental code-generation extensions for OmniFiberLM.

MTP: parallel future-token prediction with independently trained offsets.
EVO: mutation and validation of *configuration*, not model weights.
RL: clipped policy-gradient loss from verified, bounded rewards.
No claim of autonomous self-improvement or code correctness.
"""
from dataclasses import dataclass,replace
import torch
from torch import nn
import torch.nn.functional as F

class MultiTokenPrediction(nn.Module):
    """Predict next 1..K tokens from each hidden state; causal hidden states required."""
    def __init__(self,width,vocab_size,horizon=3):
        super().__init__()
        if horizon<1: raise ValueError("horizon must be positive")
        self.heads=nn.ModuleList([nn.Linear(width,vocab_size) for _ in range(horizon)])
    def forward(self,hidden):
        return [head(hidden) for head in self.heads]
    def loss(self,hidden,tokens):
        if hidden.ndim!=3 or tokens.ndim!=2 or hidden.shape[:2]!=tokens.shape:
            raise ValueError("shape mismatch")
        losses=[]
        for k,head in enumerate(self.heads,1):
            if tokens.shape[1]<=k: break
            pred=head(hidden[:,:-k])
            losses.append(F.cross_entropy(pred.reshape(-1,pred.size(-1)),tokens[:,k:].reshape(-1)))
        if not losses: raise ValueError("sequence too short")
        return torch.stack(losses).mean()

def clipped_policy_loss(log_probs,rewards,old_log_probs,clip=0.2):
    """PPO-style clipped objective. Inputs must be aligned [batch] tensors."""
    if clip<=0 or clip>=1: raise ValueError("invalid clip")
    if log_probs.shape!=rewards.shape or old_log_probs.shape!=rewards.shape:
        raise ValueError("shape mismatch")
    if not all(torch.isfinite(t).all() for t in (log_probs,rewards,old_log_probs)):
        raise ValueError("nonfinite values")
    if not (rewards.abs()<=1).all(): raise ValueError("rewards must be in [-1,1]")
    ratio=torch.exp(torch.clamp(log_probs-old_log_probs,min=-20,max=20))
    return -torch.minimum(ratio*rewards,torch.clamp(ratio,1-clip,1+clip)*rewards).mean()

@dataclass(frozen=True)
class EvoCandidate:
    lanes:int=4
    rank:int=8
    depth:int=8
    bridges:int=4

def neighboring_candidates(c):
    """Deterministic, constrained architecture mutations for paired benchmarks."""
    options=[]
    for field,step in (("lanes",1),("rank",2),("depth",2),("bridges",1)):
        for delta in (-step,step):
            value=getattr(c,field)+delta
            if value<1 and field!="bridges": continue
            if value<0 or (field=="bridges" and value>(c.lanes*(c.lanes-1))): continue
            options.append(replace(c,**{field:value}))
    return options

def bounded_code_reward(tests_passed,tests_total,security_ok,format_ok):
    """Score only from external verifier observations; no untrusted code execution."""
    if tests_total<=0 or not 0<=tests_passed<=tests_total:
        raise ValueError("invalid test counts")
    return float(max(-1.0,min(1.0,0.8*tests_passed/tests_total+0.1*bool(format_ok)+0.1*bool(security_ok)-
                                  (0.5 if not security_ok else 0.0))))
