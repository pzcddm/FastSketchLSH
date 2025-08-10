#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cminhash.h"
#include "../include/kminhash.h"
#include "../include/rminhash.h"
#include "../include/fasthash.h"
#include "../include/fasthash_simd.h"

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
                      "CMinSketch.sketch() requires all items to be integers. "
                      "Use CMinSketch for string inputs.");
                }
            }
            return self.sketch(int_items);
        }, py::arg("items"),
          "Compute MinHash signature for integer sets using CMinSketch algorithm");

    py::class_<KMinSketch>(m, "KMinSketch")
        .def( py::init<size_t, uint32_t>(),
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
                      "KMinSketch.sketch() requires string-convertible items. "
                      "Use KMinSketch for integer inputs.");
                }
            }
            return self.sketch(int_items);
        }, py::arg("items"),
          "Compute MinHash signature for integer sets using KMinSketch algorithm");

    py::class_<RMinSketch>(m, "RMinSketch")
      .def( py::init<size_t, uint32_t>(),
            py::arg("num_perm") = 128,
            py::arg("seed") = 42,
            "Initialize RMinHash with:\n"
            "  num_perm: Number of permutations (default=128)\n"
            "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

      .def("sketch", [](RMinSketch& self, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          std::vector<int> int_items;
          for (auto item : items) {
              try {
                  int_items.push_back(py::cast<int>(item));
              } catch (const py::cast_error&) {
                  throw py::value_error(
                    "RMinSketch.sketch() requires string-convertible items. "
                    "Use RMinSketch for integer inputs.");
              }
          }
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute MinHash signature for integer sets using RMinSketch algorithm");

    py::class_<FastSimilaritySketch>(m, "FastSimilaritySketch")
      .def( py::init<size_t, uint32_t>(),
            py::arg("sketch_size") = 128,
            py::arg("seed") = 42,
            "Initialize FastSimilaritySketch with:\n"
            "  sketch_size: Number of sketch\n"
            "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

      .def("sketch", [](FastSimilaritySketch& self, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          // Collect items once so we can check the first element to pick a path
          std::vector<py::object> objs;
          objs.reserve(py::len(items));
          for (auto item : items) {
              objs.emplace_back(py::reinterpret_borrow<py::object>(item));
          }
          const py::object& first = objs.front();

          const bool first_is_bytes_like = py::isinstance<py::bytes>(first)
                                        || py::isinstance<py::str>(first)
                                        || py::hasattr(first, "__bytes__");

          if (first_is_bytes_like) {
              std::vector<std::string> byte_items;
              byte_items.reserve(objs.size());
              for (const auto& obj : objs) {
                  if (py::isinstance<py::bytes>(obj)) {
                      byte_items.emplace_back(py::cast<std::string>(obj));
                  } else if (py::isinstance<py::str>(obj)) {
                      py::bytes b = py::reinterpret_borrow<py::bytes>(py::str(obj).attr("encode")("utf-8"));
                      byte_items.emplace_back(py::cast<std::string>(b));
                  } else if (py::hasattr(obj, "__bytes__")) {
                      py::bytes b = py::reinterpret_borrow<py::bytes>(obj.attr("__bytes__")());
                      byte_items.emplace_back(py::cast<std::string>(b));
                  } else {
                      throw py::value_error("All items must be bytes-like or str when the first is string-like.");
                  }
              }
              return self.sketch(byte_items);
          } else {
              std::vector<int> int_items;
              int_items.reserve(objs.size());
              for (const auto& obj : objs) {
                  try {
                      int_items.push_back(py::cast<int>(obj));
                  } catch (const py::cast_error&) {
                      throw py::value_error("All items must be integers when the first is not string-like.");
                  }
              }
              return self.sketch(int_items);
          }
      }, py::arg("items"),
        "Compute FastSimilaritySketch for: if first item is string-like (str/bytes/bytes-like) hash as bytes; otherwise cast all items to int");

    py::class_<FastSimilaritySketchAVX512Packed>(m, "FastSimilaritySketchSIMD")
      .def( py::init<size_t, uint64_t>(),
            py::arg("sketch_size") = 128,
            py::arg("seed") = 42,
            "Initialize FastSimilaritySketchSIMD with:\n"
            "  sketch_size: Number of sketch\n"
            "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

      .def("sketch", [](FastSimilaritySketchAVX512Packed& self, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          std::vector<int> int_items;
          for (auto item : items) {
              try {
                  int value = py::cast<int>(item);
                  int_items.push_back(value);
              } catch (const py::cast_error&) {
                  throw py::value_error(
                    "FastSimilaritySketchSIMD.sketch() requires all items to be integers.");
              }
          }
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute MinHash signature for integer sets using FastSimilaritySketchSIMD algorithm");
}