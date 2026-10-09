import unittest
try: import torch
except ImportError: torch=None

@unittest.skipIf(torch is None,"PyTorch optional")
class CodecTests(unittest.TestCase):
    def test_end_to_end(self):
        from dual_x.edge_codec import NanoFiberCodecLM
        m=NanoFiberCodecLM(41,width=12,rank=3,depth=2,lanes=3,edges=[(0,0,1),(1,1,2)])
        ids=torch.randint(0,41,(2,7))
        logits,meta=m(ids)
        self.assertEqual(tuple(logits.shape),(2,7,41))
        self.assertEqual(meta["bridge_count"],2)
        loss=m.next_token_loss(ids)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(m.encoder.depthwise.weight.grad)
        self.assertIsNotNone(m.decoder.refine[0].weight.grad)
    def test_causal_no_future_leak(self):
        from dual_x.edge_codec import NanoFiberCodecLM
        m=NanoFiberCodecLM(31,width=12,rank=3,depth=2,lanes=2,edges=[(0,0,1)])
        m.eval()
        a=torch.tensor([[1,2,3,4,5]])
        b=torch.tensor([[1,2,3,8,9]])
        with torch.no_grad():
            ya,_=m(a);yb,_=m(b)
        self.assertTrue(torch.allclose(ya[:,:3],yb[:,:3],atol=1e-6))
    def test_tied_embeddings(self):
        from dual_x.edge_codec import NanoFiberCodecLM
        m=NanoFiberCodecLM(20)
        self.assertIs(m.encoder.embed.weight,m.decoder.lm_head.weight)
if __name__=="__main__":unittest.main()
