import sqlite3
from decimal import Decimal

def connect(path="dual_x.sqlite3"):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS products(platform TEXT,product_id TEXT,title TEXT,price TEXT,commission_rate TEXT,affiliate_url TEXT,source TEXT,updated_at TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(platform,product_id))")
    db.execute("CREATE TABLE IF NOT EXISTS commissions(platform TEXT,order_id TEXT,product_id TEXT,amount TEXT,status TEXT,PRIMARY KEY(platform,order_id,product_id))")
    return db

def upsert_product(db,platform,product_id,title,price,rate,url,source):
    if not url.startswith("https://") or not source: raise ValueError("invalid provenance")
    if Decimal(str(price)) < 0 or not 0 <= Decimal(str(rate)) <= 1: raise ValueError("invalid economics")
    with db:
        db.execute("INSERT INTO products(platform,product_id,title,price,commission_rate,affiliate_url,source) VALUES(?,?,?,?,?,?,?) ON CONFLICT(platform,product_id) DO UPDATE SET title=excluded.title,price=excluded.price,commission_rate=excluded.commission_rate,affiliate_url=excluded.affiliate_url,source=excluded.source,updated_at=CURRENT_TIMESTAMP",(platform,product_id,title,str(price),str(rate),url,source))

def record_commission(db,platform,order_id,product_id,amount,status):
    if status not in ("pending","approved","reversed") or Decimal(str(amount)) < 0: raise ValueError("invalid commission")
    with db:
        db.execute("INSERT INTO commissions VALUES(?,?,?,?,?) ON CONFLICT(platform,order_id,product_id) DO UPDATE SET amount=excluded.amount,status=excluded.status",(platform,order_id,product_id,str(amount),status))

def approved_total(db):
    return sum((Decimal(x[0]) for x in db.execute("SELECT amount FROM commissions WHERE status='approved'")),Decimal("0"))
