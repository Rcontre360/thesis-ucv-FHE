// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;

void register_enums(py::module_& m);
void register_context(py::module_& m);
void register_keys(py::module_& m);
void register_data(py::module_& m);
void register_crypto(py::module_& m);
void register_operator(py::module_& m);

PYBIND11_MODULE(_backend, m)
{
    m.doc() =
        "HEonGPU Python bindings — CKKS scheme.\n\n"
        "Security levels: SEC128, SEC192, SEC256 (NONE is not exposed).\n\n"
        "Scale is NOT a context property; pass it to CKKSEncoder.encode() per call.\n\n"
        "Keyswitching method is inferred from the P vector size passed to\n"
        "set_coeff_modulus_bit_sizes(): one P prime -> METHOD_I, multiple -> METHOD_II.\n\n"
        "Usable multiplication levels = get_ciphertext_modulus_count() - 1.";

    register_enums(m);
    register_context(m);
    register_keys(m);
    register_data(m);
    register_crypto(m);
    register_operator(m);
}
