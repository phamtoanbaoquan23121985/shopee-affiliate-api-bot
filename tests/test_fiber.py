import unittest
from dual_x.fiber import FiberPath,route,fuse
class FiberTests(unittest.TestCase):
    def test_normalization(self):
        r=route([FiberPath("product",10,.99,2,1),FiberPath("commerce",20,.9,1,1)])
        self.assertAlmostEqual(sum(r["weights"].values()),1)
    def test_failed_path(self):
        r=route([FiberPath("a",1,0,10,5),FiberPath("b",1,1,1,0)])
        self.assertEqual(r["weights"],{"b":1})
    def test_total_outage(self):
        self.assertEqual(route([FiberPath("a",1,0,1,1)])["status"],"NO_AVAILABLE_PATH")
    def test_fusion(self):
        r=route([FiberPath("a",1,1,1,0),FiberPath("b",1,1,1,0)])
        self.assertEqual(fuse({"a":[1,3],"b":[3,1]},r),[2,2])
    def test_invalid(self):
        with self.assertRaises(ValueError):route([FiberPath("a",-1,1,1,0)])
if __name__=="__main__":unittest.main()
