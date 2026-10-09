import tempfile
import unittest
from decimal import Decimal
from dual_x.storage import connect, upsert_product, record_commission, approved_total
from dual_x.scheduler import run_job
from dual_x.web import render_products

class PlatformTests(unittest.TestCase):
    def test_settlement_and_idempotency(self):
        db = connect(":memory:")
        upsert_product(db,"shopee","p","Product","100",".1","https://example.com/p","authorized-test")
        record_commission(db,"shopee","order","p","12","pending")
        self.assertEqual(approved_total(db), Decimal("0"))
        record_commission(db,"shopee","order","p","12","approved")
        record_commission(db,"shopee","order","p","12","approved")
        self.assertEqual(approved_total(db), Decimal("12"))
        record_commission(db,"shopee","order","p","12","reversed")
        self.assertEqual(approved_total(db), Decimal("0"))
        db.close()

    def test_scheduler_retries(self):
        calls = []
        def job():
            calls.append(1)
            if len(calls) < 3: raise RuntimeError("temporary")
            return 42
        self.assertEqual(run_job(job, attempts=3, base_delay=0, sleeper=lambda _:None), 42)

    def test_xss_escaped(self):
        html = render_products([("shopee","<script>","100","https://example.com")])
        self.assertNotIn("<script>", html)
        self.assertIn("sponsored", html)
if __name__ == "__main__":
    unittest.main()
