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
        .def("sketch", [](CMinHashSketch& self, py::iterable items) {
            std::vector<int> int_items;
            for (auto item : items) {
                try {
                    int_items.push_back(py::cast<int>(item));
                } catch (const py::cast_error&) {
                    throw py::value_error("All items must be integers");
                }
            }
            return self.sketch(int_items);
        }, py::arg("items"),
          "Generate MinHash signature directly from integers");
}