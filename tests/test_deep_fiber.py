import unittest
from dual_x.deep_fiber import DeepFiberConfig,make_edges
class DeepConfigTests(unittest.TestCase):
    def test_deep_topology(self):
        c=DeepFiberConfig(depth=48,lanes=8,bridge_stride=4,bridges_per_checkpoint=3)
        edges=make_edges(c)
        self.assertEqual(len(edges),36)
        self.assertEqual(len(set(edges)),36)
        self.assertTrue(all(0<=d<48 and s!=t for d,s,t in edges))
    def test_invalid(self):
        with self.assertRaises(ValueError):make_edges(DeepFiberConfig(lanes=1,bridges_per_checkpoint=1))
    def test_empty_bridges(self):
        self.assertEqual(make_edges(DeepFiberConfig(depth=16,bridges_per_checkpoint=0)),[])
try: import torch
except ImportError: torch=None
@unittest.skipIf(torch is None,"PyTorch optional")
class DeepForwardTests(unittest.TestCase):
    def test_forward(self):
        from dual_x.deep_fiber import build_model,parameter_report
        m=build_model(DeepFiberConfig(vocab_size=32,width=12,rank=3,depth=6,lanes=3,bridge_stride=2,bridges_per_checkpoint=2))
        y,_=m(torch.randint(0,32,(2,4)))
        self.assertEqual(tuple(y.shape),(2,4,32))
        self.assertGreater(parameter_report(m)["parameters"],0)
if __name__=="__main__":unittest.main()
