import pytest

pytest.importorskip("fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_ml.ckks.context import FHEContext          # noqa: E402
from fhe_ml.utils.enums import SecurityLevel        # noqa: E402
from fhe_ml.ckks.containers.plaintext import PlaintextVector  # noqa: E402
from fhe_ml.ckks.containers.ciphertext import EncryptedVector  # noqa: E402

EPSILON = 1e-2


class TestFHEContextBuilder:
    def test_security_level_default_is_sec128(self):
        ctx = FHEContext()
        assert ctx.config.security_level == SecurityLevel.SEC128

    def test_default_builds_without_error(self):
        ctx = FHEContext.default()
        assert ctx._built is True


class TestFHEContextEncode:
    def test_encode_returns_plaintext_vector(self, built_context):
        pt = built_context.encode([1.0, 2.0, 3.0])
        assert isinstance(pt, PlaintextVector)
        assert pt.size == 3

    def test_encode_decode_roundtrip(self, built_context):
        values = [1.5, 2.5, 3.5, 4.5]
        decoded = built_context.decode(built_context.encode(values))
        assert len(decoded) == len(values)
        for expected, actual in zip(values, decoded):
            assert abs(expected - actual) < EPSILON

    def test_encode_before_build_raises(self):
        with pytest.raises(RuntimeError, match="built"):
            FHEContext().encode([1.0])


class TestFHEContextEncrypt:
    def test_encrypt_returns_encrypted_vector(self, built_context):
        ct = built_context.encrypt([1.0, 2.0])
        assert isinstance(ct, EncryptedVector)
        assert ct.size == 2

    def test_encrypt_decrypt_roundtrip(self, built_context):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = built_context.decrypt(built_context.encrypt(values))
        assert len(result) == len(values)
        for expected, actual in zip(values, result):
            assert abs(expected - actual) < EPSILON

    def test_encrypt_from_plaintext_vector(self, built_context):
        pt = built_context.encode([7.0, 8.0])
        ct = built_context.encrypt(pt)
        assert isinstance(ct, EncryptedVector)
