"""Configurable deep Nano Fiber graph, not a pretrained Fortune-scale model."""
from dataclasses import dataclass
from .edge_codec import NanoFiberCodecLM

@dataclass(frozen=True)
class DeepFiberConfig:
    vocab_size: int = 32000
    width: int = 256
    rank: int = 16
    depth: int = 32
    lanes: int = 8
    bridge_stride: int = 4
    bridges_per_checkpoint: int = 4

    def validate(self):
        if min(self.vocab_size,self.width,self.rank,self.depth,self.lanes,self.bridge_stride) < 1:
            raise ValueError("all dimensions must be positive")
        if self.bridges_per_checkpoint < 0 or self.bridges_per_checkpoint > self.lanes*(self.lanes-1):
            raise ValueError("invalid bridge budget")
        if self.lanes == 1 and self.bridges_per_checkpoint:
            raise ValueError("single lane cannot bridge")

def make_edges(config):
    config.validate()
    edges=[]
    for d in range(config.bridge_stride-1,config.depth,config.bridge_stride):
        pairs=[(s,t) for s in range(config.lanes) for t in range(config.lanes) if s!=t]
        offset=(d//config.bridge_stride)%len(pairs) if pairs else 0
        for i in range(config.bridges_per_checkpoint):
            s,t=pairs[(offset+i)%len(pairs)]
            edges.append((d,s,t))
    return edges

def build_model(config):
    config.validate()
    return NanoFiberCodecLM(
        vocab_size=config.vocab_size,width=config.width,rank=config.rank,
        depth=config.depth,lanes=config.lanes,edges=make_edges(config))

def parameter_report(model):
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters":total,"trainable_parameters":trainable,
            "estimated_fp16_weight_mib":round(total*2/1024**2,2)}
