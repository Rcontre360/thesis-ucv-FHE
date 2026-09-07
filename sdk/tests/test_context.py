import pytest

pytest.importorskip(
    "opal_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first"
)

from opal_ml.ckks.config import FHEConfig  # noqa: E402
from opal_ml.ckks.containers.ciphertext import EncryptedVector  # noqa: E402
from opal_ml.ckks.containers.plaintext import PlaintextVector  # noqa: E402
from opal_ml.ckks.context import FHEContext  # noqa: E402
from opal_ml.utils.enums import SecurityLevel  # noqa: E402

EPSILON = 1e-2


class TestFHEContextBuilder:
    def test_security_level_default_is_sec128(self):
        ctx = FHEContext(FHEConfig())
        assert ctx.config.security_level == SecurityLevel.SEC128

    def test_default_builds_without_error(self):
        ctx = FHEContext.default()
        assert ctx._ops is not None


class TestFHEContextEncode:
    def test_encode_returns_plaintext_vector(self, built_context):
        pt = built_context.encode([1.0, 2.0, 3.0])
        assert isinstance(pt, PlaintextVector)
        assert pt.size == 3

    def test_encode_decode_roundtrip(self, built_context):
        values = [1.5, 2.5, 3.5, 4.5]
        decoded = built_context.decode(built_context.encode(values))
        assert len(decoded) == len(values)
        for expected, actual in zip(values, decoded, strict=False):
            assert abs(expected - actual) < EPSILON


class TestClientEncrypt:
    def test_encrypt_returns_encrypted_vector(self, client):
        ct = client.encrypt([1.0, 2.0])
        assert isinstance(ct, EncryptedVector)
        assert ct.size == 2

    def test_encrypt_decrypt_roundtrip(self, client):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = client.decrypt(client.encrypt(values))
        assert len(result) == len(values)
        for expected, actual in zip(values, result, strict=False):
            assert abs(expected - actual) < EPSILON
