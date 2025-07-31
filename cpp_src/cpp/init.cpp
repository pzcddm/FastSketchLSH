#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cminhash.h"
#include "../include/kminhash.h"

namespace py = pybind11;

PYBIND11_MODULE(FastSketchLSH, m) {
    m.attr("__version__") = "0.2.0";

    py::class_<CMinSketch>(m, "CMinSketch")
        .def(py::init<size_t, uint32_t>(),
             py::arg("num_perm") = 128,
             py::arg("seed") = 42,
             "Initialize CMinHash with:\n"
             "  num_perm: Number of permutations (default=128)\n"
             "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")
        
        .def("sketch", [](CMinSketch& self, py::iterable items) {
            if (items.is_none() || py::len(items) == 0) {
                  throw py::value_error("Items cannot be empty");
              }
            std::vector<int> int_items;
            for (auto item : items) {
                try {
                    int_items.push_back(py::cast<int>(item));
                } catch (const py::cast_error&) {
                    throw py::value_error(
                      "CMinHashSketch.sketch() requires all items to be integers. "
                      "Use KMinHashSketch for string inputs.");
                }
            }
            return self.sketch(int_items);
        }, py::arg("items"),
          "Compute MinHash signature for integer sets using CMinHashSketch algorithm");

    py::class_<KMinSketch>(m, "KMinSketch")
        .def(py::init<size_t, uint32_t>(),
                py::arg("k") = 128,
                py::arg("seed") = 42,
                "Initialize KMinHash with:\n"
                "  k: Sketch size (default=128)\n"
                "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

        .def("sketch", [](KMinSketch& self, py::iterable items) {
            if (items.is_none() || py::len(items) == 0) {
                throw py::value_error("Items cannot be empty");
            }
            std::vector<int> int_items;
            for (auto item : items) {
                try {
                    int_items.push_back(py::cast<int>(item));
                } catch (const py::cast_error&) {
                    throw py::value_error(
                      "KMinHashSketch.sketch() requires string-convertible items. "
                      "Use KMinHashSketch for integer inputs.");
                }
            }
            return self.sketch(int_items);
        }, py::arg("items"),
          "Compute MinHash signature for integer sets using KMinHashSketch algorithm");
}