"""Tests for FHEContext — parameter validation and fluent builder."""

import pytest

from fhe_sdk.context import FHEContext
from fhe_sdk.enums import SecurityLevel


class TestFHEContextBuilder:
    def test_missing_degree_raises(self):
        with pytest.raises(ValueError, match="poly_modulus_degree"):
            FHEContext().set_coeff_modulus_bit_sizes([60, 40, 60]).set_scale(2**40).build()

    def test_missing_coeff_modulus_raises(self):
        with pytest.raises(ValueError, match="coeff_modulus_bit_sizes"):
            FHEContext().set_poly_modulus_degree(8192).set_scale(2**40).build()

    def test_missing_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            FHEContext().set_poly_modulus_degree(8192).set_coeff_modulus_bit_sizes([60, 40, 60]).build()

    def test_mutate_after_build_raises(self, built_context):
        with pytest.raises(RuntimeError, match="already built"):
            built_context.set_poly_modulus_degree(4096)

    def test_fluent_setters_return_self(self):
        ctx = FHEContext()
        assert ctx.set_poly_modulus_degree(8192) is ctx
        assert ctx.set_coeff_modulus_bit_sizes([60, 40, 60]) is ctx
        assert ctx.set_scale(2**40) is ctx
        assert ctx.set_security_level(SecurityLevel.SEC128) is ctx

    def test_security_level_default_is_sec128(self):
        ctx = FHEContext()
        assert ctx._security_level == SecurityLevel.SEC128


class TestFHEContextEncode:
    def test_encode_returns_plaintext(self, built_context):
        from fhe_sdk.plaintext import Plaintext
        pt = built_context.encode([1.0, 2.0, 3.0])
        assert isinstance(pt, Plaintext)
        assert pt.size == 3

    def test_encode_decode_roundtrip(self, built_context):
        values = [1.5, 2.5, 3.5, 4.5]
        pt = built_context.encode(values)
        decoded = built_context.decode(pt)
        assert len(decoded) == len(values)
        for expected, actual in zip(values, decoded):
            assert abs(expected - actual) < 1e-4

    def test_encode_before_build_raises(self):
        ctx = FHEContext()
        with pytest.raises(RuntimeError, match="built"):
            ctx.encode([1.0])


class TestFHEContextEncrypt:
    def test_encrypt_returns_ciphertext(self, built_context):
        from fhe_sdk.ciphertext import Ciphertext
        ct = built_context.encrypt([1.0, 2.0])
        assert isinstance(ct, Ciphertext)
        assert ct.size == 2

    def test_encrypt_decrypt_roundtrip(self, built_context):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        ct = built_context.encrypt(values)
        result = built_context.decrypt(ct)
        assert len(result) == len(values)
        for expected, actual in zip(values, result):
            assert abs(expected - actual) < 1e-3

    def test_encrypt_from_plaintext(self, built_context):
        from fhe_sdk.ciphertext import Ciphertext
        pt = built_context.encode([7.0, 8.0])
        ct = built_context.encrypt(pt)
        assert isinstance(ct, Ciphertext)


class TestCiphertextArithmetic:
    def test_add_scalar(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        result = built_context.decrypt(ct + 1.0)
        for expected, actual in zip([2.0, 3.0, 4.0], result):
            assert abs(expected - actual) < 1e-3

    def test_add_ciphertext(self, built_context):
        a = built_context.encrypt([1.0, 2.0])
        b = built_context.encrypt([3.0, 4.0])
        result = built_context.decrypt(a + b)
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < 1e-3

    def test_mul_scalar(self, built_context):
        ct = built_context.encrypt([2.0, 3.0])
        result = built_context.decrypt(ct * 2.0)
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < 1e-3

    def test_depth_mismatch_raises(self, built_context):
        ct = built_context.encrypt([1.0, 2.0])
        ct_deep = ct * 1.0  # depth 1 after rescale
        pt_fresh = built_context.encode([1.0, 1.0])  # depth 0
        with pytest.raises(ValueError, match="Depth mismatch"):
            ct_deep + pt_fresh
