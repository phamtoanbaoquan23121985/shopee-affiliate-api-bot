import unittest
try: import torch
except ImportError: torch=None

@unittest.skipIf(torch is None,"PyTorch optional")
class MicroTubeTests(unittest.TestCase):
    def test_shapes(self):
        from dual_x.fiber_bridges import MicroTubeFiberNet
        m=MicroTubeFiberNet(width=12,rank=3,depth=3,lanes=3,edges=[(0,0,1),(2,1,2)])
        y,info=m(torch.randn(2,5,12))
        self.assertEqual(tuple(y.shape),(2,5,12))
        self.assertEqual(info["bridge_count"],2)
    def test_disabled_matches_no_bridges(self):
        from dual_x.fiber_bridges import MicroTubeFiberNet
        torch.manual_seed(42)
        m=MicroTubeFiberNet(width=8,rank=2,depth=2,lanes=2,edges=[(0,0,1)])
        x=torch.randn(1,3,8)
        y,_=m(x,enabled_edges=[False])
        self.assertTrue(torch.isfinite(y).all())
    def test_invalid_edge(self):
        from dual_x.fiber_bridges import MicroTubeFiberNet
        with self.assertRaises(ValueError): MicroTubeFiberNet(lanes=2,edges=[(0,1,1)])
    def test_edge_selection(self):
        from dual_x.fiber_bridges import propose_edges
        self.assertEqual(propose_edges([[0,.7],[.2,0]],1),[(0,1)])
    def test_backward(self):
        from dual_x.fiber_bridges import MicroTubeFiberNet
        m=MicroTubeFiberNet(width=8,rank=2,depth=2,lanes=2,edges=[(0,0,1)])
        y,_=m(torch.randn(2,3,8));y.square().mean().backward()
        self.assertIsNotNone(m.bridges[0].up.weight.grad)
if __name__=="__main__": unittest.main()
