#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

using CKKSContext = HEContext<Scheme::CKKS>;
using CKKSRelinkey   = Relinkey<Scheme::CKKS>;
using CKKSGaloiskey  = Galoiskey<Scheme::CKKS>;
using CKKSPlaintext  = Plaintext<Scheme::CKKS>;
using CKKSCiphertext = Ciphertext<Scheme::CKKS>;
using CKKSEncoder    = HEEncoder<Scheme::CKKS>;
using CKKSOperator   = HEArithmeticOperator<Scheme::CKKS>;

void register_operator(py::module_& m)
{
    py::class_<CKKSOperator>(m, "CKKSOperator",
        "CKKS arithmetic operator. Performs all homomorphic operations.\n\n"
        "All in-place methods modify the first ciphertext argument and return None.\n"
        "Use ct.copy() when a new ciphertext is needed.\n\n"
        "After multiply_inplace you MUST call:\n"
        "  1. op.relinearize_inplace(ct, rk)\n"
        "  2. op.rescale_inplace(ct)\n"
        "before using ct in any further operation.\n\n"
        "Before ct+pt or ct*pt: ensure ct.depth == pt.depth.\n"
        "If they differ, call op.mod_drop_plain_inplace(pt) once per level.")

        .def(py::init([](CKKSContext& ctx, CKKSEncoder& enc) {
                 return CKKSOperator(ctx, enc);
             }),
             py::arg("context"), py::arg("encoder"),
             "Construct an operator for the given context and encoder.")

        // -------------------------------------------------------------------
        // ct OP ct
        // -------------------------------------------------------------------
        .def("add_inplace",
             [](CKKSOperator& self, CKKSCiphertext& a, CKKSCiphertext& b) {
                 self.add_inplace(a, b);
             },
             py::arg("a"), py::arg("b"),
             "a += b  (homomorphic addition, in-place).\n"
             "Preconditions: a.depth == b.depth; neither has relinearization_required.")

        .def("sub_inplace",
             [](CKKSOperator& self, CKKSCiphertext& a, CKKSCiphertext& b) {
                 self.sub_inplace(a, b);
             },
             py::arg("a"), py::arg("b"),
             "a -= b  (homomorphic subtraction, in-place). Same preconditions as add_inplace.")

        .def("negate_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct) {
                 self.negate_inplace(ct);
             },
             py::arg("ct"),
             "ct = -ct  (negate all slots, in-place). No flag preconditions.")

        .def("multiply_inplace",
             [](CKKSOperator& self, CKKSCiphertext& a, CKKSCiphertext& b) {
                 self.multiply_inplace(a, b);
             },
             py::arg("a"), py::arg("b"),
             "a *= b  (homomorphic multiplication, in-place).\n"
             "Preconditions: neither operand has rescale_required or relinearization_required.\n"
             "After this call both flags are set on a. Call relinearize_inplace then rescale_inplace.")

        // -------------------------------------------------------------------
        // ct OP pt  (Plaintext polynomial object)
        // -------------------------------------------------------------------
        .def("add_plain_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, CKKSPlaintext& pt) {
                 self.add_plain_inplace(ct, pt);
             },
             py::arg("ct"), py::arg("pt"),
             "ct += pt  (add plaintext to ciphertext, in-place).\n"
             "Preconditions: ct.depth == pt.depth; ct.relinearization_required == False.")

        .def("sub_plain_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, CKKSPlaintext& pt) {
                 self.sub_plain_inplace(ct, pt);
             },
             py::arg("ct"), py::arg("pt"),
             "ct -= pt  (subtract plaintext from ciphertext, in-place).\n"
             "Same preconditions as add_plain_inplace.")

        .def("multiply_plain_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, CKKSPlaintext& pt) {
                 self.multiply_plain_inplace(ct, pt);
             },
             py::arg("ct"), py::arg("pt"),
             "ct *= pt  (multiply ciphertext by plaintext, in-place).\n"
             "Preconditions: ct.depth == pt.depth; ct.relinearization_required == False.\n"
             "After this call ct.rescale_required = True. Does NOT set relinearization_required.")

        // -------------------------------------------------------------------
        // ct OP double  (scalar constant)
        // -------------------------------------------------------------------
        .def("add_scalar_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, double scalar) {
                 self.add_plain_inplace(ct, scalar);
             },
             py::arg("ct"), py::arg("scalar"),
             "ct += scalar  (add a real constant to every slot, in-place).\n"
             "Precondition: ct.relinearization_required == False.")

        .def("sub_scalar_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, double scalar) {
                 self.sub_plain_inplace(ct, scalar);
             },
             py::arg("ct"), py::arg("scalar"),
             "ct -= scalar  (subtract a real constant from every slot, in-place).\n"
             "Same precondition as add_scalar_inplace.")

        .def("multiply_scalar_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                double scalar, double scale) {
                 self.multiply_plain_inplace(ct, scalar, scale);
             },
             py::arg("ct"), py::arg("scalar"), py::arg("scale"),
             "ct *= scalar  (multiply every slot by a real constant, in-place).\n"
             "scale: encoding scale for the constant (typically the same as encode()).\n"
             "After this call ct.rescale_required = True.")

        // -------------------------------------------------------------------
        // Relinearize + rescale
        // -------------------------------------------------------------------
        .def("relinearize_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct, CKKSRelinkey& rk) {
                 self.relinearize_inplace(ct, rk);
             },
             py::arg("ct"), py::arg("relin_key"),
             "Reduce a size-3 ciphertext back to size-2 (in-place).\n"
             "Precondition: ct.relinearization_required == True.\n"
             "After this call ct.relinearization_required = False.")

        .def("rescale_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct) {
                 self.rescale_inplace(ct);
             },
             py::arg("ct"),
             "Drop the bottom Q prime and divide the scale by it (in-place). Consumes 1 level.\n"
             "Preconditions: ct.rescale_required == True; ct.relinearization_required == False.\n"
             "After this call ct.rescale_required = False.")

        // -------------------------------------------------------------------
        // Rotation
        // -------------------------------------------------------------------
        .def("rotate_rows_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                CKKSGaloiskey& gk, int shift) {
                 self.rotate_rows_inplace(ct, gk, shift);
             },
             py::arg("ct"), py::arg("galois_key"), py::arg("shift"),
             "Cyclically rotate the slot vector by shift positions (in-place).\n"
             "Preconditions: no pending rescale or relinearization; galois_key covers shift.")

        .def("rotate_rows",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                CKKSGaloiskey& gk, int shift) -> CKKSCiphertext {
                 CKKSCiphertext output(ct);
                 self.rotate_rows(ct, output, gk, shift);
                 return output;
             },
             py::arg("ct"), py::arg("galois_key"), py::arg("shift"),
             "Return a NEW ciphertext equal to rotate_rows(ct, shift). ct is not modified.")

        // -------------------------------------------------------------------
        // Modulus drop
        // -------------------------------------------------------------------
        .def("mod_drop_plain_inplace",
             [](CKKSOperator& self, CKKSPlaintext& pt) {
                 self.mod_drop_inplace(pt);
             },
             py::arg("pt"),
             "Drop one modulus prime from a plaintext. Increments pt.depth by 1.\n"
             "Use to align a fresh plaintext to a deeper ciphertext before ct+pt or ct*pt.")

        .def("mod_drop_inplace",
             [](CKKSOperator& self, CKKSCiphertext& ct) {
                 self.mod_drop_inplace(ct);
             },
             py::arg("ct"),
             "Drop one modulus prime from a ciphertext without consuming a multiply level.\n"
             "Preconditions: ct.rescale_required == False; ct.relinearization_required == False.")

        // -------------------------------------------------------------------
        // Bootstrapping
        //
        // Workflow:
        //   1. op.generate_bootstrapping_params(scale, config, BootstrappingType.REGULAR)
        //   2. shifts = op.bootstrapping_key_indexs()
        //   3. gk = CKKSGaloiskey(ctx, shifts); keygen.generate_galois_key(gk, sk)
        //   4. ct_fresh = op.regular_bootstrapping(ct, gk, rk)  # when ct.level == 0
        // -------------------------------------------------------------------
        .def("generate_bootstrapping_params",
             [](CKKSOperator& self, double scale,
                const BootstrappingConfig& config,
                const arithmetic_bootstrapping_type& boot_type) {
                 self.generate_bootstrapping_params(scale, config, boot_type);
             },
             py::arg("scale"), py::arg("config"), py::arg("boot_type"),
             "Precompute bootstrapping parameters. Call once after key generation.\n"
             "scale: the encoding scale used throughout computation.\n"
             "config: BootstrappingConfig (piece counts, Taylor degree, key mode).\n"
             "boot_type: BootstrappingType.REGULAR or BootstrappingType.SLIM.")

        .def("bootstrapping_key_indexs",
             [](CKKSOperator& self) -> std::vector<int> {
                 return self.bootstrapping_key_indexs();
             },
             "Return the list of Galois shift indices required for bootstrapping.\n"
             "Throws if generate_bootstrapping_params() has not been called.\n"
             "Pass this list to CKKSGaloiskey(ctx, shifts) before key generation.")

        .def("regular_bootstrapping",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                CKKSGaloiskey& gk,
                CKKSRelinkey& rk) -> CKKSCiphertext {
                 return self.regular_bootstrapping(ct, gk, rk);
             },
             py::arg("ct"), py::arg("galois_key"), py::arg("relin_key"),
             "Bootstrap ct back to a high-level ciphertext (returns new ciphertext).\n"
             "Preconditions: generate_bootstrapping_params() called; ct is at level 0;\n"
             "galois_key generated with bootstrapping_key_indexs() shift list.")

        .def("regular_bootstrapping_v2",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                CKKSGaloiskey& gk,
                CKKSRelinkey& rk) -> CKKSCiphertext {
                 return self.regular_bootstrapping_v2(ct, gk, rk,
                                                      nullptr, nullptr);
             },
             py::arg("ct"), py::arg("galois_key"), py::arg("relin_key"),
             "Non-sparse-key bootstrapping variant (IACR 2020/1203).\n"
             "Returns a new bootstrapped ciphertext. Same preconditions as regular_bootstrapping.\n"
             "Switch keys (dense<->sparse) are omitted; pass them via C++ if needed.")

        .def("slim_bootstrapping",
             [](CKKSOperator& self, CKKSCiphertext& ct,
                CKKSGaloiskey& gk,
                CKKSRelinkey& rk) -> CKKSCiphertext {
                 return self.slim_bootstrapping(ct, gk, rk);
             },
             py::arg("ct"), py::arg("galois_key"), py::arg("relin_key"),
             "Slim bootstrapping variant — fewer levels consumed than regular.\n"
             "Returns a new bootstrapped ciphertext. Same preconditions as regular_bootstrapping.");
}
