import unittest
try: import torch
except ImportError: torch=None
@unittest.skipIf(torch is None,"torch optional")
class OmniTests(unittest.TestCase):
    def test_tasks(self):
        from dual_x.omni_fiber import OmniConfig,OmniFiberLM
        c=OmniConfig(vocab_size=37,width=16,heads=4,layers=2,lanes=3,rank=4,max_context=12)
        m=OmniFiberLM(c)
        x=torch.randint(0,37,(2,5))
        expected={"language":(2,5,37),"reasoning":(2,5,37),"classification":(2,2),"retrieval":(2,16),"agent":(2,8)}
        for task,shape in expected.items():
            y,_=m(x,task)
            self.assertEqual(tuple(y.shape),shape)
        m.next_token_loss(x).backward()
        self.assertIsNotNone(m.backbone[0].attn.in_proj_weight.grad)
    def test_causality(self):
        from dual_x.omni_fiber import OmniConfig,OmniFiberLM
        m=OmniFiberLM(OmniConfig(vocab_size=20,width=16,heads=4,layers=2,lanes=2,rank=4))
        m.eval()
        with torch.no_grad():
            a,_=m(torch.tensor([[1,2,3,4]]))
            b,_=m(torch.tensor([[1,2,8,9]]))
        self.assertTrue(torch.allclose(a[:,:2],b[:,:2],atol=1e-5))
if __name__=="__main__":unittest.main()
