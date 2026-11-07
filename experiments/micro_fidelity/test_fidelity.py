import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fidelity import EdgeLUT, calibrate, exact_axis_mean, historical_classes, quiet_construct
from kan_patched import KAN


def head(width=(3, 4, 2), base_fun="silu"):
    model = nn.Module()
    model.kan = KAN(width=list(width), grid=5, k=3, seed=7, auto_save=False,
                    save_act=False, symbolic_enabled=False, base_fun=base_fun)
    model.forward = model.kan.forward
    return model.eval()


class Interaction(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = SimpleNamespace(width=[[2, 0], [1, 0]])

    def forward(self, x):
        return x.sum(1, keepdim=True).square()


class FidelityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.old, self.old_layer = historical_classes()

    def test_interaction_survives_infinite_axis_resolution(self):
        x = torch.tensor([[1., 1.]])
        f = Interaction()
        self.assertEqual(f(x).item(), 4.)
        self.assertEqual(exact_axis_mean(f, x).item(), 1.)
        lut = quiet_construct(self.old, f, grid_range=[-2, 2], grid_size=257)
        self.assertAlmostEqual(lut(x).item(), 1., places=5)

    def test_legacy_out_of_range_fades_to_zero(self):
        lut = quiet_construct(self.old, Interaction(), grid_range=[-2, 2], grid_size=17)
        self.assertEqual(lut(torch.tensor([[3., 3.]])).item(), 0.)

    def test_nonzero_intercepts_not_repeated(self):
        f = head((3, 2))
        with torch.no_grad():
            layer = f.kan.act_fun[0]
            layer.scale_base.zero_()
            layer.scale_sp.fill_(1)
            layer.coef.fill_(1)
        bounds = [(torch.full((3,), -.5), torch.full((3,), .5))]
        x = torch.zeros(5, 3)
        ref = f(x)
        self.assertTrue(torch.allclose(ref, torch.full((5, 2), 3.)))
        edge = EdgeLUT(f, bounds, 17)
        self.assertTrue(torch.allclose(edge(x), ref, atol=1e-6))
        legacy = quiet_construct(self.old_layer, layer, *bounds[0], grid_size=17)
        self.assertTrue(torch.allclose(legacy(x), ref * 3, atol=1e-5))

    def test_affines_and_masks_preserved(self):
        f = head((3, 2), "identity")
        with torch.no_grad():
            layer = f.kan.act_fun[0]
            layer.scale_sp.zero_()
            layer.scale_base.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
            layer.mask[1, 1] = 0
            f.kan.subnode_scale[0].copy_(torch.tensor([2., -1.]))
            f.kan.subnode_bias[0].copy_(torch.tensor([.5, .7]))
            f.kan.node_scale[0].copy_(torch.tensor([.4, 1.5]))
            f.kan.node_bias[0].copy_(torch.tensor([-.2, .9]))
        x = torch.tensor([[.2, .3, -.4], [-.2, .5, .4]])
        expected = ((x @ torch.tensor([[1., 2.], [3., 0.], [5., 6.]])) *
                    torch.tensor([2., -1.]) + torch.tensor([.5, .7])) * \
                    torch.tensor([.4, 1.5]) + torch.tensor([-.2, .9])
        lut = EdgeLUT(f, [(torch.full((3,), -1.), torch.ones(3))], 17)
        self.assertTrue(torch.allclose(f(x), expected, atol=1e-6))
        self.assertTrue(torch.allclose(lut(x), expected, atol=2e-6))

    def test_silu_branch_and_resolution_convergence(self):
        f = head()
        g = torch.Generator().manual_seed(90)
        cal = torch.rand(1024, 3, generator=g) * 2 - 1
        x = torch.rand(512, 3, generator=g) * 1.8 - .9
        ranges = calibrate(f, cal)
        coarse = (EdgeLUT(f, ranges, 16)(x) - f(x)).square().mean()
        fine = (EdgeLUT(f, ranges, 256)(x) - f(x)).square().mean()
        self.assertLess(fine, coarse / 100)

    def test_saturation_at_boundaries_is_explicit(self):
        f = head((3, 2))
        ranges = [(torch.full((3,), -1.), torch.ones(3))]
        lut = EdgeLUT(f, ranges, 65)
        x = torch.tensor([[2., -3., 0.]])
        self.assertTrue(torch.allclose(lut(x), f(x.clamp(-1, 1)), atol=1e-5))
        self.assertEqual(lut.layers[0].last_saturated, 2)

    def test_integer_table_storage_is_real_int8(self):
        f = head((3, 2))
        lut = EdgeLUT(f, [(torch.full((3,), -1.), torch.ones(3))], 65, True)
        self.assertEqual(lut.layers[0].table.dtype, torch.int8)
        self.assertEqual(lut.storage()["table_bytes"], 3 * 65 * 2)
        self.assertTrue(torch.isfinite(lut(torch.rand(100, 3))).all())

    def test_active_symbolic_branch_rejected(self):
        f = head((3, 2))
        f.kan.symbolic_enabled = True
        with torch.no_grad():
            f.kan.symbolic_fun[0].mask[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "symbolic"):
            EdgeLUT(f, [(torch.zeros(3), torch.ones(3))], 16)

    def test_multiplication_and_input_remapping_rejected(self):
        f = head((3, 2))
        f.kan.width[1][1] = 1
        with self.assertRaisesRegex(ValueError, "Multiplication"):
            EdgeLUT(f, [(torch.zeros(3), torch.ones(3))], 16)
        f = head((3, 2))
        f.kan.input_id = torch.tensor([2, 1, 0])
        with self.assertRaisesRegex(ValueError, "input selection"):
            EdgeLUT(f, [(torch.zeros(3), torch.ones(3))], 16)

    def test_bad_ranges_rejected(self):
        f = head((3, 2))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            EdgeLUT(f, [(torch.zeros(3), torch.zeros(3))], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
