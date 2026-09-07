from opal_ml import BootstrapConfig, Sequential, config_for_circuit
from opal_ml.layers import Linear, ReLU


def test_consumed_levels_is_ctos_plus_taylor_plus_7():
    assert BootstrapConfig(3, 3, 10).consumed_levels() == 20
    assert BootstrapConfig(4, 3, 7).consumed_levels() == 18
    # independent of stoc_piece
    assert BootstrapConfig(3, 2, 10).consumed_levels() == 20


def test_config_for_circuit_shallow_skips_bootstrap():
    cfg = config_for_circuit(scale=40, circuit_depth=6, relu_degrees=(3,))
    assert cfg.bootstrap is None
    assert cfg.coeff_modulus_bit_sizes == [60] + [40] * 6 + [60]
    assert cfg.log_n == 16
    assert cfg.log_scale == 40


def test_config_for_circuit_deep_bootstraps_maximizes_and_fits():
    cfg = config_for_circuit(scale=52, circuit_depth=200, relu_degrees=(5,) * 10)
    assert cfg.bootstrap is not None
    n_scaling = len(cfg.coeff_modulus_bit_sizes) - 2
    working_room = n_scaling - cfg.bootstrap.consumed_levels()
    assert working_room >= 3
    assert cfg.num_p() >= 2
    assert cfg.coeff_modulus_bit_sizes[0] == 60
    assert cfg.coeff_modulus_bit_sizes[-1] == 60
    assert all(p == 52 for p in cfg.coeff_modulus_bit_sizes[1:-1])


def test_generate_config_uses_circuit_depth():
    w1 = [[0.1] * 4 for _ in range(8)]
    w2 = [[0.1] * 8 for _ in range(2)]
    model = Sequential([Linear(4, 8, w1), ReLU(), Linear(8, 2, w2)])
    cfg = model.generate_config(scale=40, relu_degrees=(3,))
    # Linear(1) + ReLU((3,) -> 4) + Linear(1) = 6 scaling primes, fits without bootstrap
    assert cfg.bootstrap is None
    assert cfg.coeff_modulus_bit_sizes == [60] + [40] * 6 + [60]
