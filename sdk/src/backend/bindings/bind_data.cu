#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

using CKKSContextPtr = HEContext<Scheme::CKKS>;
using CKKSPlaintext  = Plaintext<Scheme::CKKS>;
using CKKSCiphertext = Ciphertext<Scheme::CKKS>;
using CKKSEncoder    = HEEncoder<Scheme::CKKS>;

void register_data(py::module_& m)
{
    py::class_<BootstrappingConfig>(m, "BootstrappingConfig",
        "Configuration for CKKS bootstrapping. Pass to "
        "CKKSOperator.generate_bootstrapping_params().")
        .def(py::init<int, int, int, bool>(),
             py::arg("CtoS_piece")    = 3,
             py::arg("StoC_piece")    = 3,
             py::arg("taylor_number") = 11,
             py::arg("less_key_mode") = false,
             "Construct a bootstrapping config with the given parameters.\n"
             "CtoS_piece / StoC_piece: number of pieces for CoeffToSlot / SlotToCoeff (2-5).\n"
             "taylor_number: Taylor series degree for EvalMod (default 11).\n"
             "less_key_mode: reduce Galois key count at the cost of more levels.")
        .def_readonly("CtoS_piece",    &BootstrappingConfig::CtoS_piece_)
        .def_readonly("StoC_piece",    &BootstrappingConfig::StoC_piece_)
        .def_readonly("taylor_number", &BootstrappingConfig::taylor_number_)
        .def_readonly("less_key_mode", &BootstrappingConfig::less_key_mode_);

    py::class_<CKKSPlaintext>(m, "CKKSPlaintext",
        "CKKS plaintext — an encoded (not encrypted) slot vector.\n"
        "Fill via CKKSEncoder.encode(). Before mixing with a ciphertext,\n"
        "depths must match; use CKKSOperator.mod_drop_plain_inplace(pt) per level.")

        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Allocate an uninitialised plaintext bound to the given context.")

        .def_property_readonly("depth",
             &CKKSPlaintext::depth,
             "Number of mod-drop operations applied. Fresh plaintext has depth=0.")

        .def_property_readonly("scale",
             &CKKSPlaintext::scale,
             "Encoding scale used when this plaintext was encoded.")

        .def_property_readonly("size",
             &CKKSPlaintext::size,
             "Raw polynomial size = n * (Q_size - depth).");

    py::class_<CKKSCiphertext>(m, "CKKSCiphertext",
        "CKKS ciphertext — an encrypted slot vector.\n"
        "After multiply_inplace you MUST call relinearize_inplace then rescale_inplace\n"
        "before any further operation.")

        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Allocate an uninitialised ciphertext bound to the given context.")

        .def_property_readonly("depth",
             &CKKSCiphertext::depth,
             "Number of rescale/mod_drop operations applied.")

        .def_property_readonly("level",
             &CKKSCiphertext::level,
             "Remaining usable multiplication levels = coeff_modulus_count - (depth + 1).")

        .def_property_readonly("scale",
             &CKKSCiphertext::scale,
             "Current encoding scale. Doubles after multiply_plain_inplace unless rescaled.")

        .def_property_readonly("size",
             &CKKSCiphertext::size,
             "Polynomial count: 2 normally, 3 while relinearization_required.")

        .def_property_readonly("rescale_required",
             &CKKSCiphertext::rescale_required,
             "Set after any multiplication. A subsequent ct*ct throws if this is True.\n"
             "Clear with rescale_inplace().")

        .def_property_readonly("relinearization_required",
             &CKKSCiphertext::relinearization_required,
             "Set after multiply_inplace(ct, ct). Any further multiply or add throws.\n"
             "Clear with relinearize_inplace(ct, rk).")

        .def("copy",
             [](const CKKSCiphertext& self) -> CKKSCiphertext {
                 return CKKSCiphertext(self);
             },
             "Return a device-side deep copy (cudaMemcpyAsync). "
             "Required for out-of-place arithmetic.");

    py::class_<CKKSEncoder>(m, "CKKSEncoder",
        "CKKS slot encoder/decoder. Scale is a per-encode argument.")

        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Construct an encoder for the given (already generated) context.")

        .def("encode",
             [](CKKSEncoder& self, CKKSPlaintext& pt,
                const std::vector<double>& msg, double scale) {
                 self.encode(pt, msg, scale);
             },
             py::arg("plaintext"), py::arg("message"), py::arg("scale"),
             "Encode a list of doubles into a CKKS plaintext.\n"
             "Slots beyond len(message) are padded with 0.")

        .def("decode",
             [](CKKSEncoder& self, CKKSPlaintext& pt) {
                 std::vector<double> result;
                 self.decode(result, pt);
                 return result;
             },
             py::arg("plaintext"),
             "Decode a CKKS plaintext to a list of slot_count() doubles.")

        .def("slot_count",
             &CKKSEncoder::slot_count,
             "Maximum number of plaintext slots = poly_modulus_degree / 2.");
}
