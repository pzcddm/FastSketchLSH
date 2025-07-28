#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cminhash.h"

namespace py = pybind11;

PYBIND11_MODULE(FastSketchLSH, m) {
    m.attr("__version__") = "0.2.0";

    py::class_<CMinHashSketch>(m, "CMinHashSketch")
        .def(py::init<size_t, uint32_t>(),
             py::arg("num_perm") = 128,
             py::arg("seed") = 42)
        .def("sketch", &CMinHashSketch::sketch,
             py::arg("items"),
             "Generate MinHash signature for input strings");
}