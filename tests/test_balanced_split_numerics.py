import unittest
import torch
from torch.distributions import Bernoulli, kl_divergence
from src.train_balanced_splits import bernoulli_reference_kl


class BernoulliNumericsTests(unittest.TestCase):
    def test_saturated_logits_have_finite_loss_and_gradients(self):
        reference = torch.tensor([-1000., 1000., 0., 20., -20.])
        current = torch.tensor([1000., -1000., 1000., -20., 20.], requires_grad=True)
        loss = bernoulli_reference_kl(reference, current)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue((loss >= 0).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertGreater(current.grad[0].item(), 0)
        self.assertLess(current.grad[1].item(), 0)

    def test_matches_standard_kl_away_from_saturation(self):
        reference = torch.tensor([-2., 0., 2.])
        current = torch.tensor([1., -1., .5])
        expected = kl_divergence(Bernoulli(logits=reference), Bernoulli(logits=current))
        torch.testing.assert_close(bernoulli_reference_kl(reference, current), expected)
        torch.testing.assert_close(bernoulli_reference_kl(reference, reference), torch.zeros(3))


if __name__ == '__main__':
    unittest.main()
