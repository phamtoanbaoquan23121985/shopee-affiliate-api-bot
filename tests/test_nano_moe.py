import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"PyTorch optional")
class NanoMoETests(unittest.TestCase):
    def test_shape_and_gradient(self):
        from dual_x.nano_moe import NanoFiberAgent
        m=NanoFiberAgent(40,width=24,layers=2,rank=4)
        x=torch.randint(0,40,(2,5))
        logits,routes=m(x)
        self.assertEqual(tuple(logits.shape),(2,5,40))
        self.assertEqual(len(routes),2)
        logits.square().mean().backward()
        self.assertIsNotNone(m.blocks[0].router.weight.grad)
    def test_fiber_outage(self):
        from dual_x.nano_moe import FiberMoELayer
        m=FiberMoELayer(width=12,rank=3,top_k=2)
        out,route=m(torch.randn(2,4,12),available=[True,False,True])
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse((route["indices"]==1).any())
    def test_zero_initialization(self):
        from dual_x.nano_moe import FiberMoELayer
        x=torch.randn(2,3,16)
        y,_=FiberMoELayer(width=16)(x)
        self.assertTrue(torch.equal(x,y))
if __name__=="__main__":unittest.main()
