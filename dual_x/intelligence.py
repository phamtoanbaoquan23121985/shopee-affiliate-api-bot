"""Evidence-gated affiliate revenue and repeat-purchase measurements."""
from dataclasses import dataclass
from decimal import Decimal as D

@dataclass(frozen=True)
class Cohort:
    platform: str
    product_id: str
    clicks: int
    approved_commission: D
    pending_commission: D = D("0")
    reversed_commission: D = D("0")
    ad_spend: D = D("0")
    approved_orders: int = 0
    repeat_buyers: int = 0
    eligible_buyers: int = 0
    observed_days: int = 0

def measure(c, min_clicks=100, min_eligible=30, min_days=30):
    if c.clicks < 0 or c.approved_orders < 0 or c.eligible_buyers < 0 or c.repeat_buyers < 0 or c.repeat_buyers > c.eligible_buyers:
        raise ValueError("invalid cohort counts")
    if any(x < 0 or not x.is_finite() for x in (c.approved_commission,c.pending_commission,c.reversed_commission,c.ad_spend)):
        raise ValueError("invalid financial amounts")
    if c.observed_days < 0 or min_clicks < 1 or min_eligible < 1 or min_days < 1:
        raise ValueError("invalid observation window")
    epc = c.approved_commission * D(1000) / D(c.clicks) if c.clicks else None
    net = (c.approved_commission - c.ad_spend) * D(1000) / D(c.clicks) if c.clicks else None
    repeat = D(c.repeat_buyers) / D(c.eligible_buyers) if c.eligible_buyers else None
    ready = c.clicks >= min_clicks and c.eligible_buyers >= min_eligible and c.observed_days >= min_days
    return {"platform":c.platform,"product_id":c.product_id,
            "approved_epc_1000":str(epc) if epc is not None else None,
            "net_1000":str(net) if net is not None else None,
            "repeat_rate":str(repeat) if repeat is not None else None,
            "status":"OBSERVED_ELIGIBLE" if ready else "INSUFFICIENT_OBSERVATIONS",
            "pending_excluded":str(c.pending_commission),"reversed_excluded":str(c.reversed_commission)}

def rank(cohorts):
    records = [measure(c) for c in cohorts]
    return sorted(records,key=lambda x:(x["status"]=="OBSERVED_ELIGIBLE",D(x["net_1000"]) if x["net_1000"] is not None else D("-Infinity")),reverse=True)
