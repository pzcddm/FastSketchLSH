#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cminhash.h"
#include "../include/kminhash.h"
#include "../include/rminhash.h"
#include "../include/fasthash.h"
#include "../include/fasthash_simd.h"
#include "../include/fastsketch_lsh.h"
#include "../include/fastsketch_rensa_lsh.h"

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
          std::vector<uint32_t> int_items;
          for (auto item : items) {
              try {
                  uint32_t value = py::cast<uint32_t>(item);
                  int_items.push_back(value);
              } catch (const py::cast_error&) {
                  throw py::value_error(
                    "FastSimilaritySketchSIMD.sketch() requires all items to be integers.");
              }
          }
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute MinHash signature for integer sets using FastSimilaritySketchSIMD algorithm");

    py::class_<FastSketchLSH>(m, "FastSketchLSH")
      .def(py::init<float, size_t, size_t, uint32_t>(),
            py::arg("threshold"),
            py::arg("sketch_size"),
            py::arg("bands"),
            py::arg("random_seed") = 42,
            "Initialize FastSketchLSH with:\n"
            "  threshold: Jaccard similarity threshold (0 < threshold < 1)\n"
            "  sketch_size: Length of sketch vector (must be divisible by bands)\n"
            "  bands: Number of bands to split sketch into\n"
            "  random_seed: Random seed (default=42)")
      
      .def("insert", [](FastSketchLSH& self, const std::string& key, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          
          // Check the type of the first element to decide which overload to use
          auto first_item = *items.begin();
          if (py::isinstance<py::str>(first_item) || py::isinstance<py::bytes>(first_item)) {
              std::vector<std::string> str_items;
              for (auto item : items) {
                  if (py::isinstance<py::str>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else if (py::isinstance<py::bytes>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else {
                      throw py::value_error("All items must be strings or bytes when first item is string-like");
                  }
              }
              self.insert(key, str_items);
          } else {
              std::vector<int> int_items;
              for (auto item : items) {
                  try {
                      int_items.push_back(py::cast<int>(item));
                  } catch (const py::cast_error&) {
                      throw py::value_error("All items must be integers when first item is not string-like");
                  }
              }
              self.insert(key, int_items);
          }
      }, py::arg("key"), py::arg("items"),
          "Insert a set with a key into LSH index\n"
          "Automatically detects whether items are strings or integers")
      
      .def("query", [](FastSketchLSH& self, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          
          // Check the type of the first element to decide which overload to use
          auto first_item = *items.begin();
          if (py::isinstance<py::str>(first_item) || py::isinstance<py::bytes>(first_item)) {
              std::vector<std::string> str_items;
              for (auto item : items) {
                  if (py::isinstance<py::str>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else if (py::isinstance<py::bytes>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else {
                      throw py::value_error("All items must be strings or bytes when first item is string-like");
                  }
              }
              return self.query(str_items);
          } else {
              std::vector<int> int_items;
              for (auto item : items) {
                  try {
                      int_items.push_back(py::cast<int>(item));
                  } catch (const py::cast_error&) {
                      throw py::value_error("All items must be integers when first item is not string-like");
                  }
              }
              return self.query(int_items);
          }
      }, py::arg("items"),
          "Query LSH index for similar sets\n"
          "Automatically detects whether items are strings or integers")
      
      .def("remove", &FastSketchLSH::remove, py::arg("key"),
            "Remove a key from the LSH index")
      
      .def("clear", &FastSketchLSH::clear,
            "Remove all keys from the LSH index");


    py::class_<FastSketchLSHRensa>(m, "FastSketchLSHRensa")
      // 构造函数
      .def(py::init<double, std::size_t, std::size_t>(),
           py::arg("threshold"),
           py::arg("num_perm"),
           py::arg("num_bands"),
           "Initialize FastSketchLSHRensa with:\n"
           "  threshold: Jaccard similarity threshold (0 < threshold < 1)\n"
           "  num_perm : Number of permutations (sketch length)\n"
           "  num_bands: Number of bands (must divide num_perm)")

      .def("insert",
           [](FastSketchLSHRensa& self, std::size_t key, const FastSimilaritySketch& sketch) {
               self.insert(key, sketch);
           },
           py::arg("key"), py::arg("sketch"),
           "Insert a sketch with a numeric key")

      .def("query",
           [](const FastSketchLSHRensa& self, const FastSimilaritySketch& sketch) {
               return self.query(sketch);
           },
           py::arg("sketch"),
           "Query candidates for a given sketch")

      // 只读属性
      .def_property_readonly("num_perm",  &FastSketchLSHRensa::num_perm)
      .def_property_readonly("num_bands", &FastSketchLSHRensa::num_bands)
      .def_property_readonly("band_size",&FastSketchLSHRensa::band_size)
      .def_property_readonly("threshold",&FastSketchLSHRensa::threshold);
}