import numpy as np
import pytest

from fhe_ml.layers.relu import ReLU, _fn_coeffs


# Exact rational coefficients for the first few f_n (ePrint 2019/417, Table 1).
# Format: array indexed by power; even-power slots are zero.
F1_EXACT = np.array([0.0, 3 / 2, 0.0, -1 / 2])
F2_EXACT = np.array([0.0, 15 / 8, 0.0, -10 / 8, 0.0, 3 / 8])
F3_EXACT = np.array([0.0, 35 / 16, 0.0, -35 / 16, 0.0, 21 / 16, 0.0, -5 / 16])


class TestFnCoeffsClosedForm:
    def test_degree_3_matches_textbook(self):
        np.testing.assert_allclose(_fn_coeffs(3), F1_EXACT, atol=1e-12)

    def test_degree_5_matches_textbook(self):
        np.testing.assert_allclose(_fn_coeffs(5), F2_EXACT, atol=1e-12)

    def test_degree_7_matches_textbook(self):
        np.testing.assert_allclose(_fn_coeffs(7), F3_EXACT, atol=1e-12)


class TestFnCoeffsInvariants:
    @pytest.mark.parametrize("degree", [3, 5, 7, 9, 15])
    def test_odd_polynomial_has_zero_even_terms(self, degree):
        coeffs = _fn_coeffs(degree)
        assert coeffs.shape == (degree + 1,)
        np.testing.assert_array_equal(coeffs[0::2], np.zeros((degree + 1) // 2))

    @pytest.mark.parametrize("degree", [3, 5, 7, 9, 15])
    def test_value_at_one_is_one(self, degree):
        coeffs = _fn_coeffs(degree)
        value = np.polynomial.polynomial.polyval(1.0, coeffs)
        assert abs(value - 1.0) < 1e-10

    @pytest.mark.parametrize("degree", [3, 5, 7, 9, 15])
    def test_derivatives_at_one_vanish(self, degree):
        # f_n^(j)(1) = 0 for j = 1..n, where n = (degree - 1) // 2.
        coeffs = _fn_coeffs(degree)
        poly = np.polynomial.Polynomial(coeffs)
        n = (degree - 1) // 2
        for j in range(1, n + 1):
            value = poly.deriv(j)(1.0)
            assert abs(value) < 1e-8, f"f_{n}^({j})(1) = {value}, expected 0"

    @pytest.mark.parametrize("degree", [3, 5, 7, 9, 15])
    def test_oddness_symmetry(self, degree):
        # An odd polynomial satisfies p(-x) = -p(x); check on a grid.
        coeffs = _fn_coeffs(degree)
        grid = np.linspace(-1.0, 1.0, 21)
        values_pos = np.polynomial.polynomial.polyval(grid, coeffs)
        values_neg = np.polynomial.polynomial.polyval(-grid, coeffs)
        np.testing.assert_allclose(values_neg, -values_pos, atol=1e-12)

    def test_even_degree_rejected(self):
        with pytest.raises(ValueError, match="odd and >= 3"):
            _fn_coeffs(4)

    def test_degree_one_rejected(self):
        with pytest.raises(ValueError, match="odd and >= 3"):
            _fn_coeffs(1)


def _relu(degrees):
    r = ReLU()
    r.set_degrees(degrees)
    return r


class TestReLUForwardPlain:
    def test_single_polynomial_approximates_relu_far_from_kink(self):
        relu = _relu((3,))
        # Inputs far from 0 — where polynomial approximation is well-behaved.
        x = np.array([-1.0, -0.8, 0.8, 1.0])
        expected = np.maximum(0.0, x)
        out = relu.forward_plain(x)
        assert np.max(np.abs(out - expected)) < 0.15

    def test_composed_polynomials_tighter_than_single(self):
        # f_n composition is monotonic in sharpness: deg=(3,3) beats (3,) on
        # the same off-kink grid.
        single = _relu((3,))
        composed = _relu((3, 3))
        x = np.linspace(-1.0, -0.4, 30)  # negative side; expected ReLU=0.
        err_single = np.max(np.abs(single.forward_plain(x)))
        err_composed = np.max(np.abs(composed.forward_plain(x)))
        assert err_composed < err_single

    def test_high_degree_chain_is_very_accurate(self):
        relu = _relu((15, 15, 27))
        x = np.linspace(-1.0, 1.0, 101)
        # Skip the kink window where polynomial approximation is intrinsically weak.
        mask = np.abs(x) > 0.1
        out = relu.forward_plain(x[mask])
        expected = np.maximum(0.0, x[mask])
        assert np.max(np.abs(out - expected)) < 0.01

    def test_shape_preserved(self):
        relu = _relu((3,))
        x = np.array([[0.5, -0.5], [0.7, -0.7]])
        assert relu.forward_plain(x).shape == x.shape

    def test_forward_calibration_is_real_relu(self):
        # The calibration path must use the true (non-polynomial) ReLU.
        relu = _relu((3,))
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        np.testing.assert_array_equal(
            relu.forward_calibration(x),
            np.maximum(0.0, x),
        )


class TestReLUConfig:
    def test_empty_degrees_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            ReLU().set_degrees(())

    def test_unset_degrees_raises_on_call(self):
        # ReLU() leaves coeffs unresolved; forward_plain must refuse rather
        # than silently misbehave.
        relu = ReLU()
        with pytest.raises(RuntimeError, match="degrees unresolved"):
            relu.forward_plain(np.array([1.0]))

    def test_mult_depth_matches_formula(self):
        # (d+3)/2 per polynomial + 1 final x*step. Same formula as test_relu.py
        # but checked directly here so this file is self-contained.
        assert _relu((3,)).mult_depth() == 4
        assert _relu((3, 3)).mult_depth() == 7
        assert _relu((5,)).mult_depth() == 5
        assert _relu((7,)).mult_depth() == 6
