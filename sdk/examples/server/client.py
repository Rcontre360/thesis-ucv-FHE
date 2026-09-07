from collections.abc import Callable
from opal_ml import Client, EncryptedVector, FHEContext


def setup_client(context: FHEContext) -> Client:
    return Client(context)


def send_request(
    client: Client,
    server: Callable[[EncryptedVector], EncryptedVector],
    values: list[float],
) -> list[float]:
    encrypted = client.encrypt(values)
    encrypted_result = server(encrypted)
    return client.decrypt(encrypted_result)
