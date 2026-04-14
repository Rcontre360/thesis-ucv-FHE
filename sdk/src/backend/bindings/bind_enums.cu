#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <heongpu/heongpu.hpp>

namespace py = pybind11;
using namespace heongpu;

void register_enums(py::module_& m)
{
    py::enum_<sec_level_type>(m, "SecurityLevel",
        "Post-quantum security level for CKKS context construction.\n"
        "Pass to create_ckks_context_with_security(). NONE is not exposed.")
        .value("SEC128", sec_level_type::sec128,
               "128-bit post-quantum security (default).")
        .value("SEC192", sec_level_type::sec192,
               "192-bit post-quantum security.")
        .value("SEC256", sec_level_type::sec256,
               "256-bit post-quantum security.");

    py::enum_<arithmetic_bootstrapping_type>(m, "BootstrappingType",
        "Bootstrapping algorithm variant. Pass to generate_bootstrapping_params().")
        .value("REGULAR", arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING,
               "Standard CKKS bootstrapping.")
        .value("SLIM", arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING,
               "Slim bootstrapping (fewer levels consumed).");
}
