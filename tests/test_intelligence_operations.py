import sqlite3
import unittest
from decimal import Decimal as D
from dual_x.intelligence import Cohort, measure, rank
from dual_x.operations import initialize, create_draft, approve, mark_published, cancel

class RevenueTests(unittest.TestCase):
    def test_approved_only(self):
        c=Cohort("shopee","a",1000,D("100"),D("999"),D("200"),D("20"),10,4,40,35)
        m=measure(c)
        self.assertEqual(m["approved_epc_1000"],"100")
        self.assertEqual(m["net_1000"],"80")
        self.assertEqual(m["repeat_rate"],"0.1")
        self.assertEqual(m["status"],"OBSERVED_ELIGIBLE")
    def test_insufficient(self):
        self.assertEqual(measure(Cohort("lazada","b",0,D("0")))["status"],"INSUFFICIENT_OBSERVATIONS")
    def test_rank(self):
        high=Cohort("shopee","a",1000,D("100"),eligible_buyers=40,observed_days=40)
        weak=Cohort("lazada","b",10,D("200"),eligible_buyers=1,observed_days=1)
        self.assertEqual(rank([weak,high])[0]["product_id"],"a")
    def test_invalid(self):
        with self.assertRaises(ValueError): measure(Cohort("shopee","a",10,D("-1")))

class OperationTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(":memory:");initialize(self.db)
    def tearDown(self): self.db.close()
    def test_approval_gate(self):
        create_draft(self.db,"c1","shopee","p1","site")
        with self.assertRaises(ValueError): mark_published(self.db,"c1")
        approve(self.db,"c1","merchant")
        mark_published(self.db,"c1")
        self.assertEqual(self.db.execute("SELECT status FROM campaigns").fetchone()[0],"published")
    def test_cancel(self):
        create_draft(self.db,"c2","lazada","p2","site");cancel(self.db,"c2")
        with self.assertRaises(ValueError): approve(self.db,"c2","merchant")
if __name__=="__main__": unittest.main()
