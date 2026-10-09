import unittest
try: import torch
except ImportError: torch=None

@unittest.skipIf(torch is None,"torch optional")
class TerminalFiberTests(unittest.TestCase):
    def test_shape(self):
        from dual_x.terminal_fiber import TerminalFiberLanguageHead
        model=TerminalFiberLanguageHead(32,width=16,rank=4,depth=2)
        y,info=model(torch.randint(0,32,(2,5)))
        self.assertEqual(tuple(y.shape),(2,5,32))
        self.assertEqual(tuple(info["lanes"].shape),(2,5,3,16))
    def test_independent_lanes(self):
        from dual_x.terminal_fiber import TerminalFiberLayer
        m=TerminalFiberLayer(width=8,rank=3,depth=2)
        x=torch.randn(2,3,8)
        before=m.lanes[1](x).detach().clone()
        with torch.no_grad(): m.lanes[0][0].up.weight.fill_(1)
        after=m.lanes[1](x)
        self.assertTrue(torch.equal(before,after))
    def test_terminal_only_and_fallback(self):
        from dual_x.terminal_fiber import TerminalFiberLayer
        m=TerminalFiberLayer(width=8,rank=3)
        _,info=m(torch.randn(1,2,8),available=[False,True,False])
        self.assertTrue(torch.equal(info["terminal_weights"][...,1],torch.ones(1,2)))
    def test_grad(self):
        from dual_x.terminal_fiber import TerminalFiberLayer
        m=TerminalFiberLayer(width=8,rank=3)
        y,_=m(torch.randn(2,3,8))
        y.square().mean().backward()
        self.assertIsNotNone(m.lanes[0][0].up.weight.grad)
if __name__=="__main__":unittest.main()
