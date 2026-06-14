import numpy as np
import pytest

pytest.importorskip(
    "fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first"
)

from fhe_ml.ckks.containers.tensor import PlaintextTensor  # noqa: E402

EPSILON = 1e-2


class TestEncryptedVectorArithmetic:
    def test_add_scalar(self, built_context, client):
        ct = client.encrypt([1.0, 2.0, 3.0])
        result = client.decrypt(ct + 1.0)
        for expected, actual in zip([2.0, 3.0, 4.0], result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_add_ciphertext(self, built_context, client):
        a = client.encrypt([1.0, 2.0])
        b = client.encrypt([3.0, 4.0])
        result = client.decrypt(a + b)
        for expected, actual in zip([4.0, 6.0], result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_mul_scalar(self, built_context, client):
        ct = client.encrypt([2.0, 3.0])
        result = client.decrypt(ct * 2.0)
        for expected, actual in zip([4.0, 6.0], result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_mul_ciphertext(self, built_context, client):
        a = client.encrypt([2.0, 3.0])
        b = client.encrypt([4.0, 5.0])
        result = client.decrypt(a * b)
        for expected, actual in zip([8.0, 15.0], result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_sub_ciphertext(self, built_context, client):
        a = client.encrypt([5.0, 6.0])
        b = client.encrypt([3.0, 2.0])
        result = client.decrypt(a - b)
        for expected, actual in zip([2.0, 4.0], result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_depth_mismatch_raises(self, built_context, client):
        ct = client.encrypt([1.0, 2.0])
        ct_deep = ct * 1.0
        pt_fresh = built_context.encode([1.0, 1.0])
        with pytest.raises(ValueError, match="Depth mismatch"):
            _ = ct_deep + pt_fresh


class TestEncryptedVectorRotate:
    def test_rotate_zero_is_identity(self, built_context, client):
        client.generate_rotation_keys([1])
        built_context.set_client_params(client.key_params())
        values = [1.0, 2.0, 3.0, 4.0]
        ct = client.encrypt(values)
        result = client.decrypt(ct.rotate(0))[:4]
        for expected, actual in zip(values, result, strict=False):
            assert abs(expected - actual) < EPSILON

    def test_rotate_shifts_values(self, built_context, client):
        client.generate_rotation_keys([1])
        built_context.set_client_params(client.key_params())
        ct = client.encrypt([1.0, 2.0, 3.0, 4.0])
        result = client.decrypt(ct.rotate(1))
        assert abs(result[0] - 2.0) < EPSILON
        assert abs(result[1] - 3.0) < EPSILON
        assert abs(result[2] - 4.0) < EPSILON


class TestEncryptedVectorMatmul:
    def test_matmul_identity(self, built_context, client):
        x = client.encrypt([3.0, 5.0])
        W = PlaintextTensor([[1.0, 0.0], [0.0, 1.0]])
        W.encode(built_context)
        client.generate_rotation_keys(W.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        result = x.matmul(W)
        assert result.size == 2
        dec = client.decrypt(result)
        assert abs(dec[0] - 3.0) < EPSILON
        assert abs(dec[1] - 5.0) < EPSILON

    def test_matmul_projection(self, built_context, client):
        x = client.encrypt([2.0, 4.0, 6.0])
        W = PlaintextTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        W.encode(built_context)
        client.generate_rotation_keys(W.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        result = x.matmul(W)
        assert result.size == 2
        dec = client.decrypt(result)
        assert abs(dec[0] - 2.0) < EPSILON
        assert abs(dec[1] - 4.0) < EPSILON

    def test_matmul_shape_mismatch_raises(self, built_context, client):
        x = client.encrypt([1.0, 2.0, 3.0])
        W = PlaintextTensor([[1.0, 0.0], [0.0, 1.0]])
        W.encode(built_context)
        client.generate_rotation_keys(W.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        with pytest.raises(ValueError, match="columns"):
            x.matmul(W)

    def test_matmul_wrong_type_raises(self, built_context, client):
        x = client.encrypt([1.0, 2.0])
        with pytest.raises(TypeError, match="PlaintextTensor"):
            x.matmul([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_matmul_tall_expansion(self, built_context, client):
        # out > in: rectangular cyclic-wrap matmul handles the tall case
        # natively — no padding, no level cost from period extension.
        x = client.encrypt([1.0, 2.0, 3.0, 4.0])
        W = PlaintextTensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        W.encode(built_context)
        client.generate_rotation_keys(W.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        result = x.matmul(W)
        assert result.size == 8
        dec = client.decrypt(result)
        expected = [1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 7.0, 5.0]
        for e, a in zip(expected, dec, strict=False):
            assert abs(e - a) < EPSILON

    def test_matmul_chain_wide_then_narrow(self, built_context, client):
        # Chained matmul where the second layer is narrower than the first.
        # Under replicated I/O this works without any tile-period bookkeeping.
        x = client.encrypt([1.0, 2.0, 3.0, 4.0])
        W1 = PlaintextTensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        W1.encode(built_context)
        client.generate_rotation_keys(W1.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        y = x.matmul(W1)  # size=2 (extracts x[0], x[1])
        W2 = PlaintextTensor([[1.0, 1.0]])  # 1x2: out=1, in=2 (sum)
        W2.encode(built_context)
        client.generate_rotation_keys(W2.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        z = y.matmul(W2)  # size=1
        assert z.size == 1
        assert abs(client.decrypt(z)[0] - 3.0) < EPSILON

    def test_matmul_dense_32x16_vs_numpy(self, built_context, client):
        rng = np.random.default_rng(seed=20250101)
        x_np = rng.uniform(-1.0, 1.0, size=16)
        W_np = rng.uniform(-1.0, 1.0, size=(32, 16))
        expected = W_np @ x_np

        x = client.encrypt(x_np.tolist())
        W = PlaintextTensor(W_np.tolist())
        W.encode(built_context)
        client.generate_rotation_keys(W.bsgs_shifts())
        built_context.set_client_params(client.key_params())
        result = x.matmul(W)
        assert result.size == 32

        dec = np.asarray(client.decrypt(result))
        assert np.max(np.abs(dec - expected)) < 5e-2
