from opal_ml import FHEConfig, FHEContext, SecurityLevel
from client import send_request, setup_client
from server import REDUCTION_SHIFTS, start_server

INPUT: list[float] = [1, 2, 3, 4, 5, 6, 7, 8]

config = FHEConfig(
    log_n=12,
    coeff_modulus_bit_sizes=[60, 40, 40, 40, 60],
    log_scale=40,
    security_level=SecurityLevel.NONE,
)
config.set_galois_shifts(REDUCTION_SHIFTS)
context = FHEContext(config)

client = setup_client(context)
server = start_server(context, client.key_params())

print("SENT:")
print(INPUT)
result = send_request(client, server, INPUT)
print("RETURNED:")
print(round(result[0], 4))
