import csv
from decimal import Decimal

class AuthorizedFeedAdapter:
    """Import only product exports explicitly authorized for your account."""
    def __init__(self,platform,path):
        if platform not in ("shopee","lazada"): raise ValueError("unsupported platform")
        self.platform,self.path=platform,path

    def products(self):
        with open(self.path,encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                if row["platform"].lower()!=self.platform: continue
                price=Decimal(row["price"])
                rate=Decimal(row["commission_rate"])
                if price<0 or not 0<=rate<=1: raise ValueError("invalid feed")
                if not row["affiliate_url"].startswith("https://"): raise ValueError("invalid link")
                yield type("Product",(),dict(platform=self.platform,product_id=row["product_id"],title=row["title"],price=str(price),commission_rate=str(rate),affiliate_url=row["affiliate_url"],provenance="authorized-csv:"+self.path))()

    def commissions(self):
        raise NotImplementedError("Commission reports require verified settlement mapping")
