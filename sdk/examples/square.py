"""Minimal CKKS example: encrypt a vector, square it on GPU, decrypt."""

from fhe_sdk import (
    create_ckks_context,
    CKKSKeyGenerator,
    CKKSSecretkey,
    CKKSPublickey,
    CKKSRelinkey,
    CKKSEncoder,
    CKKSEncryptor,
    CKKSDecryptor,
    CKKSOperator,
    CKKSPlaintext,
    CKKSCiphertext,
)

# 1. Create CKKS context
ctx = create_ckks_context()
ctx.set_poly_modulus_degree(8192)
ctx.set_coeff_modulus_bit_sizes([60, 30, 30, 30], [60])
ctx.generate()
ctx.print_parameters()

# 2. Generate keys
keygen = CKKSKeyGenerator(ctx)
secret_key = CKKSSecretkey(ctx)
keygen.generate_secret_key(secret_key)

public_key = CKKSPublickey(ctx)
keygen.generate_public_key(public_key, secret_key)

relin_key = CKKSRelinkey(ctx)
keygen.generate_relin_key(relin_key, secret_key)

# 3. Setup crypto pipeline
encoder = CKKSEncoder(ctx)
encryptor = CKKSEncryptor(ctx, public_key)
decryptor = CKKSDecryptor(ctx, secret_key)
ops = CKKSOperator(ctx, encoder)

# 4. Encode and encrypt
scale = 2.0**30
slot_count = ctx.get_poly_modulus_degree() // 2
message = [10.0, 20.0, 30.0, 40.0, 0.5] + [3.0] * (slot_count - 5)

print(f"Input (first 5):    {message[:5]}")

pt = CKKSPlaintext(ctx)
encoder.encode(pt, message, scale)

ct = CKKSCiphertext(ctx)
encryptor.encrypt(ct, pt)

# 5. Homomorphic square on GPU
ops.multiply_inplace(ct, ct)
ops.relinearize_inplace(ct, relin_key)
ops.rescale_inplace(ct)

# 6. Decrypt and decode
pt_result = CKKSPlaintext(ctx)
decryptor.decrypt(pt_result, ct)
result = encoder.decode(pt_result)

print(f"Result (first 5):   {[round(x, 2) for x in result[:5]]}")
print(f"Expected:           [100.0, 400.0, 900.0, 1600.0, 0.25]")
