#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

// Alias the CKKS specializations for readability
using CKKSContext = HEContextImpl<Scheme::CKKS>;
using CKKSContextPtr = HEContext<Scheme::CKKS>;
using CKKSSecretkey = Secretkey<Scheme::CKKS>;
using CKKSPublickey = Publickey<Scheme::CKKS>;
using CKKSRelinkey = Relinkey<Scheme::CKKS>;
using CKKSPlaintext = Plaintext<Scheme::CKKS>;
using CKKSCiphertext = Ciphertext<Scheme::CKKS>;
using CKKSEncoder = HEEncoder<Scheme::CKKS>;
using CKKSKeygen = HEKeyGenerator<Scheme::CKKS>;
using CKKSEncryptor = HEEncryptor<Scheme::CKKS>;
using CKKSDecryptor = HEDecryptor<Scheme::CKKS>;
using CKKSOperator = HEArithmeticOperator<Scheme::CKKS>;

PYBIND11_MODULE(_heongpu_backend, m) {
    m.doc() = "HEonGPU Python bindings for GPU-accelerated CKKS";

    // --- Enums ---
    py::enum_<Scheme>(m, "Scheme")
        .value("BFV", Scheme::BFV)
        .value("CKKS", Scheme::CKKS)
        .value("TFHE", Scheme::TFHE)
        .export_values();

    py::enum_<sec_level_type>(m, "SecurityLevel")
        .value("NONE", sec_level_type::none)
        .value("SEC128", sec_level_type::sec128)
        .value("SEC192", sec_level_type::sec192)
        .value("SEC256", sec_level_type::sec256)
        .export_values();

    py::enum_<keyswitching_type>(m, "KeyswitchingType")
        .value("NONE", keyswitching_type::NONE)
        .value("METHOD_I", keyswitching_type::KEYSWITCHING_METHOD_I)
        .value("METHOD_II", keyswitching_type::KEYSWITCHING_METHOD_II)
        .export_values();

    // --- CKKS Context ---
    py::class_<CKKSContext, CKKSContextPtr>(m, "CKKSContext")
        .def("set_poly_modulus_degree", &CKKSContext::set_poly_modulus_degree)
        .def("set_coeff_modulus_bit_sizes", &CKKSContext::set_coeff_modulus_bit_sizes)
        .def("generate", py::overload_cast<>(&CKKSContext::generate))
        .def("print_parameters", &CKKSContext::print_parameters)
        .def("get_poly_modulus_degree", &CKKSContext::get_poly_modulus_degree)
        .def("get_ciphertext_modulus_count", &CKKSContext::get_ciphertext_modulus_count);

    m.def("create_ckks_context", []() -> CKKSContextPtr {
        return GenHEContext<Scheme::CKKS>();
    }, "Create a new CKKS context");

    // --- Keys ---
    py::class_<CKKSSecretkey>(m, "CKKSSecretkey")
        .def(py::init<CKKSContextPtr>());

    py::class_<CKKSPublickey>(m, "CKKSPublickey")
        .def(py::init<CKKSContextPtr>());

    py::class_<CKKSRelinkey>(m, "CKKSRelinkey")
        .def(py::init<CKKSContextPtr>());

    // --- KeyGenerator ---
    py::class_<CKKSKeygen>(m, "CKKSKeyGenerator")
        .def(py::init<CKKSContextPtr>())
        .def("generate_secret_key", [](CKKSKeygen& self, CKKSSecretkey& sk) {
            self.generate_secret_key(sk);
        })
        .def("generate_public_key", [](CKKSKeygen& self, CKKSPublickey& pk, CKKSSecretkey& sk) {
            self.generate_public_key(pk, sk);
        })
        .def("generate_relin_key", [](CKKSKeygen& self, CKKSRelinkey& rk, CKKSSecretkey& sk) {
            self.generate_relin_key(rk, sk);
        });

    // --- Plaintext / Ciphertext ---
    py::class_<CKKSPlaintext>(m, "CKKSPlaintext")
        .def(py::init<CKKSContextPtr>());

    py::class_<CKKSCiphertext>(m, "CKKSCiphertext")
        .def(py::init<CKKSContextPtr>());

    // --- Encoder ---
    py::class_<CKKSEncoder>(m, "CKKSEncoder")
        .def(py::init<CKKSContextPtr>())
        .def("encode", [](CKKSEncoder& self, CKKSPlaintext& pt,
                          const std::vector<double>& msg, double scale) {
            self.encode(pt, msg, scale);
        })
        .def("decode", [](CKKSEncoder& self, CKKSPlaintext& pt) {
            std::vector<double> result;
            self.decode(result, pt);
            return result;
        });

    // --- Encryptor / Decryptor ---
    py::class_<CKKSEncryptor>(m, "CKKSEncryptor")
        .def(py::init<CKKSContextPtr, CKKSPublickey&>())
        .def("encrypt", [](CKKSEncryptor& self, CKKSCiphertext& ct, CKKSPlaintext& pt) {
            self.encrypt(ct, pt);
        });

    py::class_<CKKSDecryptor>(m, "CKKSDecryptor")
        .def(py::init<CKKSContextPtr, CKKSSecretkey&>())
        .def("decrypt", [](CKKSDecryptor& self, CKKSPlaintext& pt, CKKSCiphertext& ct) {
            self.decrypt(pt, ct);
        });

    // --- Arithmetic Operator ---
    py::class_<CKKSOperator>(m, "CKKSOperator")
        .def(py::init<CKKSContextPtr, CKKSEncoder&>())
        .def("add_inplace", [](CKKSOperator& self, CKKSCiphertext& a, CKKSCiphertext& b) {
            self.add_inplace(a, b);
        })
        .def("multiply_inplace", [](CKKSOperator& self, CKKSCiphertext& a, CKKSCiphertext& b) {
            self.multiply_inplace(a, b);
        })
        .def("relinearize_inplace", [](CKKSOperator& self, CKKSCiphertext& ct, CKKSRelinkey& rk) {
            self.relinearize_inplace(ct, rk);
        })
        .def("rescale_inplace", [](CKKSOperator& self, CKKSCiphertext& ct) {
            self.rescale_inplace(ct);
        });
}
