#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <iostream>

#include "fideslib.hpp"

namespace py = pybind11;
using namespace fideslib;

PYBIND11_MODULE(fideslib_python, m) {
    m.doc() = "FIDESlib Python bindings for GPU-accelerated CKKS";

    // Enums
    py::enum_<PKESchemeFeature>(m, "PKESchemeFeature")
        .value("PKE", PKESchemeFeature::PKE)
        .value("KEYSWITCH", PKESchemeFeature::KEYSWITCH)
        .value("PRE", PKESchemeFeature::PRE)
        .value("LEVELEDSHE", PKESchemeFeature::LEVELEDSHE)
        .value("ADVANCEDSHE", PKESchemeFeature::ADVANCEDSHE)
        .value("MULTIPARTY", PKESchemeFeature::MULTIPARTY)
        .value("FHE", PKESchemeFeature::FHE)
        .value("SCHEMESWITCH", PKESchemeFeature::SCHEMESWITCH)
        .export_values();

    py::enum_<ScalingTechnique>(m, "ScalingTechnique")
        .value("FIXEDMANUAL", ScalingTechnique::FIXEDMANUAL)
        .value("FIXEDAUTO", ScalingTechnique::FIXEDAUTO)
        .value("FLEXIBLEAUTO", ScalingTechnique::FLEXIBLEAUTO)
        .value("FLEXIBLEAUTOEXT", ScalingTechnique::FLEXIBLEAUTOEXT)
        .export_values();

    // Plaintext
    py::class_<PlaintextImpl, std::shared_ptr<PlaintextImpl>>(m, "Plaintext")
        .def(py::init<>());

    // Ciphertext
    py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext")
        .def(py::init<>());

    // KeyPair
    py::class_<KeyPair<DCRTPoly>>(m, "KeyPair")
        .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
        .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);

    // CCParams
    py::class_<CCParams<CryptoContextCKKSRNS>, std::shared_ptr<CCParams<CryptoContextCKKSRNS>>>(m, "CCParamsCKKSRNS")
        .def(py::init<>())
        .def("SetMultiplicativeDepth", &CCParams<CryptoContextCKKSRNS>::SetMultiplicativeDepth)
        .def("SetScalingModSize", &CCParams<CryptoContextCKKSRNS>::SetScalingModSize)
        .def("SetBatchSize", &CCParams<CryptoContextCKKSRNS>::SetBatchSize)
        .def("SetRingDim", &CCParams<CryptoContextCKKSRNS>::SetRingDim)
        .def("SetScalingTechnique", &CCParams<CryptoContextCKKSRNS>::SetScalingTechnique)
        .def("SetFirstModSize", &CCParams<CryptoContextCKKSRNS>::SetFirstModSize)
        .def("SetSecurityLevel", &CCParams<CryptoContextCKKSRNS>::SetSecurityLevel)
        .def("SetDevices", [](CCParams<CryptoContextCKKSRNS> &self, std::vector<int> devices) {
            self.SetDevices(std::move(devices));
        });

    // CryptoContext
    py::class_<CryptoContextImpl<DCRTPoly>, std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(m, "CryptoContext")
        .def(py::init<>())
        .def("Enable", static_cast<void (CryptoContextImpl<DCRTPoly>::*)(PKESchemeFeature)>(&CryptoContextImpl<DCRTPoly>::Enable))
        .def("GetRingDimension", &CryptoContextImpl<DCRTPoly>::GetRingDimension)
        .def("SetDevices", &CryptoContextImpl<DCRTPoly>::SetDevices)
        .def("LoadContext", &CryptoContextImpl<DCRTPoly>::LoadContext)
        .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen)
        .def("EvalMultKeyGen", &CryptoContextImpl<DCRTPoly>::EvalMultKeyGen)
        .def("EvalRotateKeyGen", &CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen)
        .def("MakeCKKSPackedPlaintext", static_cast<Plaintext (CryptoContextImpl<DCRTPoly>::*)(const std::vector<double>&, size_t, uint32_t, std::shared_ptr<void>, uint32_t)>(&CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext),
            py::arg("value"), py::arg("noiseScaleDeg") = 1, py::arg("level") = 0, py::arg("params") = nullptr, py::arg("slots") = 0)
        .def("Encrypt", static_cast<Ciphertext<DCRTPoly> (CryptoContextImpl<DCRTPoly>::*)(Plaintext&, const PublicKey<DCRTPoly>&)>(&CryptoContextImpl<DCRTPoly>::Encrypt))
        .def("EvalAdd", static_cast<Ciphertext<DCRTPoly> (CryptoContextImpl<DCRTPoly>::*)(const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&)>(&CryptoContextImpl<DCRTPoly>::EvalAdd))
        .def("EvalMult", static_cast<Ciphertext<DCRTPoly> (CryptoContextImpl<DCRTPoly>::*)(const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&)>(&CryptoContextImpl<DCRTPoly>::EvalMult))
        .def("EvalRotate", &CryptoContextImpl<DCRTPoly>::EvalRotate)
        .def("Rescale", &CryptoContextImpl<DCRTPoly>::Rescale)
        .def("Synchronize", &CryptoContextImpl<DCRTPoly>::Synchronize);

    // GenCryptoContext
    m.def("GenCryptoContext", &GenCryptoContext, "Generate a FIDESlib CryptoContext from parameters");
}
