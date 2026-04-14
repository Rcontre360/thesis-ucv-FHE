#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

using CKKSContextPtr = HEContext<Scheme::CKKS>;
using CKKSPublickey  = Publickey<Scheme::CKKS>;
using CKKSSecretkey  = Secretkey<Scheme::CKKS>;
using CKKSPlaintext  = Plaintext<Scheme::CKKS>;
using CKKSCiphertext = Ciphertext<Scheme::CKKS>;
using CKKSEncryptor  = HEEncryptor<Scheme::CKKS>;
using CKKSDecryptor  = HEDecryptor<Scheme::CKKS>;

void register_crypto(py::module_& m)
{
    py::class_<CKKSEncryptor>(m, "CKKSEncryptor",
        "CKKS encryptor. Encrypts CKKSPlaintext objects using the public key.")

        .def(py::init<CKKSContextPtr, CKKSPublickey&>(),
             py::arg("context"), py::arg("public_key"),
             "Construct an encryptor from a generated context and public key.")

        .def("encrypt",
             [](CKKSEncryptor& self, CKKSCiphertext& ct, CKKSPlaintext& pt) {
                 self.encrypt(ct, pt);
             },
             py::arg("ciphertext"), py::arg("plaintext"),
             "Encrypt plaintext into ciphertext under the stored public key.\n"
             "ciphertext must be a pre-allocated CKKSCiphertext(ctx).");

    py::class_<CKKSDecryptor>(m, "CKKSDecryptor",
        "CKKS decryptor. Decrypts CKKSCiphertext objects using the secret key.")

        .def(py::init<CKKSContextPtr, CKKSSecretkey&>(),
             py::arg("context"), py::arg("secret_key"),
             "Construct a decryptor from a generated context and secret key.")

        .def("decrypt",
             [](CKKSDecryptor& self, CKKSPlaintext& pt, CKKSCiphertext& ct) {
                 self.decrypt(pt, ct);
             },
             py::arg("plaintext"), py::arg("ciphertext"),
             "Decrypt ciphertext into plaintext.\n"
             "Follow with encoder.decode(plaintext) to recover float values.");
}
