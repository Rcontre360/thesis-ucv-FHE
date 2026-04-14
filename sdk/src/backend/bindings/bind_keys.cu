#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

using CKKSContextPtr = HEContext<Scheme::CKKS>;
using CKKSSecretkey  = Secretkey<Scheme::CKKS>;
using CKKSPublickey  = Publickey<Scheme::CKKS>;
using CKKSRelinkey   = Relinkey<Scheme::CKKS>;
using CKKSGaloiskey  = Galoiskey<Scheme::CKKS>;
using CKKSKeygen     = HEKeyGenerator<Scheme::CKKS>;

void register_keys(py::module_& m)
{
    py::class_<CKKSSecretkey>(m, "CKKSSecretkey",
        "CKKS secret key. Construct with the context, then fill via "
        "CKKSKeyGenerator.generate_secret_key(sk).")
        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Allocate an empty secret key bound to the given context.");

    py::class_<CKKSPublickey>(m, "CKKSPublickey",
        "CKKS public key. Construct with the context, then fill via "
        "CKKSKeyGenerator.generate_public_key(pk, sk).")
        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Allocate an empty public key bound to the given context.");

    py::class_<CKKSRelinkey>(m, "CKKSRelinkey",
        "CKKS relinearization key. Required after every multiply_inplace().\n"
        "Construct with the context, then fill via "
        "CKKSKeyGenerator.generate_relin_key(rk, sk).")
        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Allocate an empty relin key bound to the given context.");

    py::class_<CKKSGaloiskey>(m, "CKKSGaloiskey",
        "CKKS Galois (rotation) key.\n"
        "Three construction modes:\n"
        "  CKKSGaloiskey(ctx)             — powers-of-2 shifts up to MAX_SHIFT\n"
        "  CKKSGaloiskey(ctx, max_shift)  — all shifts in (-2^max_shift, 2^max_shift)\n"
        "  CKKSGaloiskey(ctx, [s0,s1,...]) — exact list of required shifts\n"
        "Fill via CKKSKeyGenerator.generate_galois_key(gk, sk).")

        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Galois key covering powers-of-2 rotation shifts up to MAX_SHIFT.")

        .def(py::init<CKKSContextPtr, int>(),
             py::arg("context"), py::arg("max_shift"),
             "Galois key covering all rotations in (-2^max_shift, 2^max_shift).")

        .def(py::init(
                 [](CKKSContextPtr ctx, const std::vector<int>& shifts) {
                     // HEonGPU's shift-vector constructor takes a non-const ref, so we copy.
                     std::vector<int> s = shifts;
                     return CKKSGaloiskey(ctx, s);
                 }),
             py::arg("context"), py::arg("shifts"),
             "Galois key for exactly the rotation steps in the given list.\n"
             "Use for bootstrapping: pass bootstrapping_key_indexs() as the shift list.");

    py::class_<CKKSKeygen>(m, "CKKSKeyGenerator",
        "Key generator for CKKS. Fills pre-allocated key objects from a secret key.")

        .def(py::init<CKKSContextPtr>(),
             py::arg("context"),
             "Construct a key generator for an already-generated context.")

        .def("generate_secret_key",
             [](CKKSKeygen& self, CKKSSecretkey& sk) {
                 self.generate_secret_key(sk);
             },
             py::arg("sk"),
             "Fill sk with a freshly sampled ternary secret key.")

        .def("generate_public_key",
             [](CKKSKeygen& self, CKKSPublickey& pk, CKKSSecretkey& sk) {
                 self.generate_public_key(pk, sk);
             },
             py::arg("pk"), py::arg("sk"),
             "Fill pk with the public key derived from sk.")

        .def("generate_relin_key",
             [](CKKSKeygen& self, CKKSRelinkey& rk, CKKSSecretkey& sk) {
                 self.generate_relin_key(rk, sk);
             },
             py::arg("rk"), py::arg("sk"),
             "Fill rk with the relinearization key derived from sk.\n"
             "The keyswitching method (I or II) is set at context generate() time.")

        .def("generate_galois_key",
             [](CKKSKeygen& self, CKKSGaloiskey& gk, CKKSSecretkey& sk) {
                 self.generate_galois_key(gk, sk);
             },
             py::arg("gk"), py::arg("sk"),
             "Fill gk with rotation keys for all shifts registered in gk.\n"
             "The shift set is fixed at Galoiskey construction time.");
}
