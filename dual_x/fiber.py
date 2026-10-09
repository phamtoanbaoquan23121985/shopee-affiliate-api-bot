"""Subsea-fiber-inspired multi-path message routing; no claim of physical optical networking."""
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Sequence

@dataclass(frozen=True)
class FiberPath:
    name: str
    latency_ms: float
    reliability: float
    capacity: float
    signal: float

def route(paths: Sequence[FiberPath], temperature: float = 1.0, min_reliability: float = 0.0):
    """Capacity-aware soft routing with failure fallback and normalized weights."""
    if not paths or not isfinite(temperature) or temperature <= 0:
        raise ValueError("paths required and temperature must be positive")
    for p in paths:
        if not all(isfinite(x) for x in (p.latency_ms,p.reliability,p.capacity,p.signal)):
            raise ValueError("nonfinite path parameter")
        if p.latency_ms < 0 or not 0 <= p.reliability <= 1 or p.capacity < 0:
            raise ValueError("invalid path")
    eligible = [p for p in paths if p.reliability >= min_reliability and p.capacity > 0 and p.reliability > 0]
    if not eligible:
        return {"status":"NO_AVAILABLE_PATH","weights":{}}
    logits = [(p.signal + log(p.reliability) + log(p.capacity) - log(1+p.latency_ms))/temperature for p in eligible]
    offset = max(logits)
    values = [exp(x-offset) for x in logits]
    z = sum(values)
    return {"status":"ROUTED","weights":{p.name:v/z for p,v in zip(eligible,values)}}

def fuse(vectors: dict[str,Sequence[float]], routing: dict):
    """Weighted fusion of product, behavioral and economics embeddings."""
    if routing["status"] != "ROUTED":
        raise ValueError("no viable path")
    weights = routing["weights"]
    if any(name not in vectors for name in weights):
        raise ValueError("missing vector")
    lengths = {len(vectors[name]) for name in weights}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("dimension mismatch")
    if any(not isfinite(x) for name in weights for x in vectors[name]):
        raise ValueError("nonfinite embedding")
    return [sum(w*vectors[name][i] for name,w in weights.items()) for i in range(next(iter(lengths)))]
