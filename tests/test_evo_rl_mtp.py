import unittest
from dual_x.evo_rl_mtp import EvoCandidate,neighboring_candidates,bounded_code_reward
class EvoTests(unittest.TestCase):
    def test_mutations(self):
        self.assertGreater(len(neighboring_candidates(EvoCandidate())),0)
    def test_reward(self):
        self.assertLess(bounded_code_reward(0,4,False,False),bounded_code_reward(4,4,True,True))
    def test_no_tests(self):
        with self.assertRaises(ValueError):bounded_code_reward(0,0,True,True)
try: import torch
except ImportError: torch=None
@unittest.skipIf(torch is None,"torch optional")
class RLAndMTPTests(unittest.TestCase):
    def test_mtp_backward(self):
        from dual_x.evo_rl_mtp import MultiTokenPrediction
        m=MultiTokenPrediction(12,23,3)
        h=torch.randn(2,6,12,requires_grad=True)
        loss=m.loss(h,torch.randint(0,23,(2,6)))
        loss.backward()
        self.assertIsNotNone(h.grad)
    def test_rl_gradient(self):
        from dual_x.evo_rl_mtp import clipped_policy_loss
        logp=torch.tensor([-.4,-.8],requires_grad=True)
        loss=clipped_policy_loss(logp,torch.tensor([1.,-.5]),torch.tensor([-.5,-.7]))
        loss.backward()
        self.assertTrue(torch.isfinite(logp.grad).all())
if __name__=="__main__":unittest.main()
