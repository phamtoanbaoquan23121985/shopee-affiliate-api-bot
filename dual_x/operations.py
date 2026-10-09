"""Human-approved campaign lifecycle. No automatic third-party posting."""
import sqlite3
from datetime import datetime, timezone

SCHEMA = """CREATE TABLE IF NOT EXISTS campaigns (
 id TEXT PRIMARY KEY, platform TEXT NOT NULL, product_id TEXT NOT NULL,
 destination TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('draft','approved','published','cancelled')),
 approval_actor TEXT, approval_at TEXT, published_at TEXT);"""

def initialize(db: sqlite3.Connection):
    db.execute(SCHEMA)

def create_draft(db, campaign_id, platform, product_id, destination):
    if platform not in ("shopee","lazada") or not all((campaign_id,product_id,destination)):
        raise ValueError("invalid campaign")
    with db:
        db.execute("INSERT INTO campaigns(id,platform,product_id,destination,status) VALUES(?,?,?,?,?)",
                   (campaign_id,platform,product_id,destination,"draft"))

def approve(db, campaign_id, actor):
    if not actor: raise ValueError("approval actor required")
    with db:
        result = db.execute("UPDATE campaigns SET status='approved',approval_actor=?,approval_at=? WHERE id=? AND status='draft'",
                            (actor,datetime.now(timezone.utc).isoformat(),campaign_id))
        if result.rowcount != 1: raise ValueError("campaign not in draft state")

def mark_published(db, campaign_id):
    """Call only after separately confirmed publication by a permitted integration."""
    with db:
        result = db.execute("UPDATE campaigns SET status='published',published_at=? WHERE id=? AND status='approved'",
                            (datetime.now(timezone.utc).isoformat(),campaign_id))
        if result.rowcount != 1: raise ValueError("campaign not approved")

def cancel(db, campaign_id):
    with db:
        result = db.execute("UPDATE campaigns SET status='cancelled' WHERE id=? AND status IN ('draft','approved')",(campaign_id,))
        if result.rowcount != 1: raise ValueError("campaign cannot be cancelled")
