#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

using CKKSContext    = HEContextImpl<Scheme::CKKS>;
using CKKSContextPtr = HEContext<Scheme::CKKS>;

void register_context(py::module_& m)
{
    py::class_<CKKSContext, CKKSContextPtr>(m, "CKKSContext",
        "CKKS context. Use create_ckks_context() or "
        "create_ckks_context_with_security() to construct.\n"
        "Usable multiplication levels = get_ciphertext_modulus_count() - 1.")

        .def("set_poly_modulus_degree",
             &CKKSContext::set_poly_modulus_degree,
             py::arg("degree"),
             "Set the polynomial ring degree (power of 2). Call before generate().")

        .def("set_coeff_modulus_bit_sizes",
             [](CKKSContext& self,
                const std::vector<int>& q_bit_sizes,
                const std::vector<int>& p_bit_sizes) {
                 self.set_coeff_modulus_bit_sizes(q_bit_sizes, p_bit_sizes);
             },
             py::arg("q_bit_sizes"), py::arg("p_bit_sizes"),
             "Set Q and P prime bit-size vectors.\n"
             "One P prime -> KEYSWITCHING_METHOD_I; multiple -> METHOD_II.\n"
             "Total Q+P bits must not exceed the security-level limit.")

        .def("set_coeff_modulus_bit_sizes_flat",
             [](CKKSContext& self, const std::vector<int>& bit_sizes) {
                 if (bit_sizes.size() < 2) {
                     throw std::invalid_argument(
                         "bit_sizes must contain at least 2 elements");
                 }
                 std::vector<int> q_bits(bit_sizes.begin(), bit_sizes.end() - 1);
                 std::vector<int> p_bits = {bit_sizes.back()};
                 self.set_coeff_modulus_bit_sizes(q_bits, p_bits);
             },
             py::arg("bit_sizes"),
             "Convenience overload: flat list where the last element is the single P prime.\n"
             "Equivalent to set_coeff_modulus_bit_sizes(bit_sizes[:-1], [bit_sizes[-1]]).\n"
             "Raises ValueError if len(bit_sizes) < 2.")

        .def("generate",
             py::overload_cast<>(&CKKSContext::generate),
             "Validate parameters and build NTT tables.\n"
             "Call after set_poly_modulus_degree() and set_coeff_modulus_bit_sizes().\n"
             "Throws std::runtime_error if total Q+P bits exceed the security-level limit.")

        .def("print_parameters",
             &CKKSContext::print_parameters,
             "Print a human-readable summary of context parameters to stdout.")

        .def("get_poly_modulus_degree",
             &CKKSContext::get_poly_modulus_degree,
             "Return the polynomial ring degree N.")

        .def("get_ciphertext_modulus_count",
             &CKKSContext::get_ciphertext_modulus_count,
             "Return the number of primes in Q. Usable levels = this value - 1.")

        .def("get_key_modulus_count",
             &CKKSContext::get_key_modulus_count,
             "Return total prime count in Q' = Q + P.");

    m.def("create_ckks_context",
          []() -> CKKSContextPtr {
              return GenHEContext<Scheme::CKKS>();
          },
          "Create a CKKS context with the default 128-bit security level.");

    m.def("create_ckks_context_with_security",
          [](sec_level_type sec) -> CKKSContextPtr {
              return GenHEContext<Scheme::CKKS>(sec);
          },
          py::arg("security_level"),
          "Create a CKKS context with the given SecurityLevel.\n"
          "Security level is a constructor argument and cannot be changed after creation.");
}
