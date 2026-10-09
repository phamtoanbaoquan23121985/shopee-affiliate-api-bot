"""Evidence-aware affiliate product ranking. No API credentials required."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass(frozen=True)
class Product:
    platform: str
    product_id: str
    price: Decimal
    commission_rate: Decimal
    conversion_rate: Optional[Decimal] = None
    repeat_probability: Optional[Decimal] = None
    repeat_orders: int = 0
    ad_cost_per_click: Decimal = Decimal("0")
    refund_rate: Decimal = Decimal("0")

def evaluate(p: Product, clicks: int = 1000) -> dict:
    if clicks < 0 or p.repeat_orders < 0:
        raise ValueError("counts must be nonnegative")
    for name in ("commission_rate", "refund_rate"):
        value = getattr(p, name)
        if not Decimal("0") <= value <= Decimal("1"):
            raise ValueError(f"{name} must be between 0 and 1")
    if p.price < 0 or p.ad_cost_per_click < 0:
        raise ValueError("price and ad cost must be nonnegative")
    for name in ("conversion_rate", "repeat_probability"):
        value = getattr(p, name)
        if value is not None and not Decimal("0") <= value <= Decimal("1"):
            raise ValueError(f"{name} must be between 0 and 1")
    commission = p.price * p.commission_rate
    if p.conversion_rate is None:
        return {"product_id": p.product_id, "platform": p.platform,
                "commission_per_order": str(commission),
                "expected_net_per_1000_clicks": None, "status": "INSUFFICIENT_DATA"}
    repeat = p.repeat_probability if p.repeat_probability is not None else Decimal("0")
    orders_per_conversion = Decimal("1") + repeat * p.repeat_orders
    gross = Decimal(clicks) * p.conversion_rate * commission * (Decimal("1") - p.refund_rate) * orders_per_conversion
    net = gross - Decimal(clicks) * p.ad_cost_per_click
    return {"product_id": p.product_id, "platform": p.platform,
            "commission_per_order": str(commission),
            "expected_net_per_1000_clicks": str(net),
            "status": "ESTIMATED" if p.repeat_probability is not None else "ESTIMATED_NO_REPEAT_DATA"}
