/*
  FastSketchLSH Python↔C++ boundary optimizations (2025-08):

  - NumPy zero-copy fast paths (numeric):
    RMinSketch::sketch(np.int32),
    FastSimilaritySketch::sketch(np.uint32 | np.int32),
    FastSketchLSH::insert/query(np.int32).
    Requirements: 1-D arrays. Reading uses buffer access (no per-element boxing),
    compute runs under GIL release.

  - Bytes/memoryview fast paths (text/bytes-like):
    FastSimilaritySketch::{sketch(list[bytes]), sketch_buffers(list[memoryview|bytes])},
    FastSketchLSH::{insert, query}(list[bytes] | list[str]).
    list[bytes]/memoryview uses PyBytes_AsStringAndSize or buffer protocol to avoid copies.
    list[str] remains supported (back-compat); bytes is fastest.

  - GIL release:
    All compute-heavy code paths (sketch/insert/query, including LSHRensa) release the GIL.

  - Backward compatibility:
    Iterable overloads are preserved. Numeric Python lists and string lists still work, though
    slower than NumPy/bytes fast paths.

  - Guidance:
    Prefer NumPy arrays (np.int32/np.uint32) for numbers and list[bytes]/memoryview for text
    to hit the fast paths.

  - Windows compatibility:
    Adds ssize_t typedef and buffer handling fixes.

  - Deprecated:
    Scalar legacy implementation preserved as FastSimilaritySketchDeprecated (C++ only);
    not exposed to Python.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <cstddef>  // For size_t and ssize_t
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#include "../include/rminhash.h"
#include "../include/fastsketch.h"
#include "../include/fastsketch_lsh.h"
#include "../include/fastsketch_rensa_lsh.h"

namespace py = pybind11;

// ===================== Optimized Helper Functions =====================

// Fast path for NumPy arrays - zero copy access
template<typename T>
inline std::vector<T> numpy_to_vector_zerocopy(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw py::value_error("NumPy array must be 1-dimensional");
    }
    T* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.size);
}

// Fast path for bytes objects - zero copy access
inline std::vector<std::string> bytes_list_to_vector_zerocopy(py::list items) {
    std::vector<std::string> result;
    result.reserve(items.size());
    
    for (auto item : items) {
        if (py::isinstance<py::bytes>(item)) {
            // Zero-copy access to bytes data
            char* data = nullptr;
            ssize_t size = 0;
            if (PyBytes_AsStringAndSize(item.ptr(), &data, &size) == -1) {
                throw py::value_error("Failed to extract bytes data");
            }
            result.emplace_back(data, size);
        } else {
            throw py::value_error("All items must be bytes objects for fast path");
        }
    }
    return result;
}

// Buffer protocol support for memoryview/bytes
inline std::vector<std::string> buffer_list_to_vector_zerocopy(py::list items) {
    std::vector<std::string> result;
    result.reserve(items.size());
    
    for (auto item : items) {
        if (py::isinstance<py::bytes>(item)) {
            char* data = nullptr;
            ssize_t size = 0;
            if (PyBytes_AsStringAndSize(item.ptr(), &data, &size) == -1) {
                throw py::value_error("Failed to extract bytes data");
            }
            result.emplace_back(data, size);
        } else if (py::isinstance<py::memoryview>(item) || py::hasattr(item, "__buffer__")) {
            try {
                py::object obj = py::reinterpret_borrow<py::object>(item);
                py::buffer_info buf = py::buffer(obj).request();
                if (buf.format != "B" && buf.format != "b") {
                    throw py::value_error("Buffer must contain bytes (format 'B' or 'b')");
                }
                if (buf.ndim != 1) {
                    throw py::value_error("Buffer must be 1-dimensional");
                }
                char* data = static_cast<char*>(buf.ptr);
                result.emplace_back(data, buf.size);
            } catch (const std::exception&) {
                throw py::value_error("Failed to extract buffer data");
            }
        } else {
            throw py::value_error("All items must be bytes or buffer objects for fast path");
        }
    }
    return result;
}

PYBIND11_MODULE(FastSketchLSH, m) {
    m.attr("__version__") = "0.2.0";

    // CMinSketch removed (deprecated)

    py::class_<RMinSketch>(m, "RMinSketch")
      .def( py::init<size_t, uint32_t>(),
            py::arg("num_perm") = 128,
            py::arg("seed") = 42,
            "Initialize RMinHash with:\n"
            "  num_perm: Number of permutations (default=128)\n"
            "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

      // Optimized NumPy array sketch method (zero-copy, GIL release)
      .def("sketch", [](RMinSketch& self, py::array_t<int32_t> arr) {
          if (arr.size() == 0) {
              throw py::value_error("Array cannot be empty");
          }
          std::vector<int> int_items = numpy_to_vector_zerocopy<int32_t>(arr);
          py::gil_scoped_release release;
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute MinHash signature for NumPy int32 array (optimized zero-copy)")

      // Fallback iterable sketch method (backward compatibility)
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
          py::gil_scoped_release release;
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute MinHash signature for integer sets using RMinSketch algorithm");

    // Note: FastSimilaritySketch (scalar) bindings have been deprecated and removed from Python.

    py::class_<FastSimilaritySketch>(m, "FastSimilaritySketch")
      .def( py::init<size_t, uint64_t>(),
            py::arg("sketch_size") = 128,
            py::arg("seed") = 42,
            "Initialize FastSimilaritySketch with:\n"
            "  sketch_size: Number of sketch\n"
            "  seed: Random seed (0 to 0xFFFFFFFF, default=42)")

      // Optimized NumPy array sketch method for uint32 (zero-copy, GIL release)
      .def("sketch", [](FastSimilaritySketch& self, py::array_t<uint32_t> arr) {
          if (arr.size() == 0) {
              throw py::value_error("Array cannot be empty");
          }
          std::vector<uint32_t> int_items = numpy_to_vector_zerocopy<uint32_t>(arr);
          py::gil_scoped_release release;
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute FastSimilaritySketchSIMD for NumPy uint32 array (optimized zero-copy)")

      // Optimized NumPy array sketch method for int32 (zero-copy, GIL release)
      .def("sketch", [](FastSimilaritySketch& self, py::array_t<int32_t> arr) {
          if (arr.size() == 0) {
              throw py::value_error("Array cannot be empty");
          }
          auto int32_items = numpy_to_vector_zerocopy<int32_t>(arr);
          std::vector<uint32_t> int_items;
          int_items.reserve(int32_items.size());
          for (int32_t val : int32_items) {
              if (val < 0) {
                  throw py::value_error("FastSimilaritySketchSIMD requires non-negative integers");
              }
              int_items.push_back(static_cast<uint32_t>(val));
          }
          py::gil_scoped_release release;
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute FastSimilaritySketchSIMD for NumPy int32 array (optimized zero-copy, converted to uint32)")

      // Optimized list sketch method (supports both bytes and strings)
      .def("sketch", [](FastSimilaritySketch& self, py::list items) {
          if (items.size() == 0) {
              throw py::value_error("List cannot be empty");
          }
          
          // Check if first item is bytes for fast path
          auto first_item = items[0];
          if (py::isinstance<py::bytes>(first_item)) {
              // Fast path for bytes objects
              std::vector<std::string> byte_items = bytes_list_to_vector_zerocopy(items);
              py::gil_scoped_release release;
              return self.sketch(byte_items);
          } else if (py::isinstance<py::str>(first_item)) {
              // Handle string lists (backward compatibility)
              std::vector<std::string> str_items;
              str_items.reserve(items.size());
              for (auto item : items) {
                  if (py::isinstance<py::str>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else if (py::isinstance<py::bytes>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else {
                      throw py::value_error("All items must be strings or bytes");
                  }
              }
              py::gil_scoped_release release;
              return self.sketch(str_items);
          } else {
              throw py::value_error("Use sketch(numpy_array) for integers or ensure all items are strings/bytes for this overload");
          }
      }, py::arg("items"),
        "Compute FastSimilaritySketchSIMD for list of strings/bytes (optimized for bytes)")

      // Buffer protocol sketch method for memoryview/bytes (zero-copy, GIL release)
      .def("sketch_buffers", [](FastSimilaritySketch& self, py::list items) {
          if (items.size() == 0) {
              throw py::value_error("List cannot be empty");
          }
          std::vector<std::string> byte_items = buffer_list_to_vector_zerocopy(items);
          py::gil_scoped_release release;
          return self.sketch(byte_items);
      }, py::arg("items"),
        "Compute FastSimilaritySketchSIMD for list of buffer objects (memoryview/bytes, optimized zero-copy)")

      // Fallback iterable sketch method (backward compatibility)
      .def("sketch", [](FastSimilaritySketch& self, py::iterable items) {
          if (items.is_none() || py::len(items) == 0) {
              throw py::value_error("Items cannot be empty");
          }
          // Inspect the first element to decide path
          std::vector<py::object> objs; objs.reserve(py::len(items));
          for (auto item : items) { objs.emplace_back(py::reinterpret_borrow<py::object>(item)); }
          const py::object& first = objs.front();
          const bool first_is_bytes_like = py::isinstance<py::bytes>(first)
                                        || py::isinstance<py::str>(first)
                                        || py::hasattr(first, "__bytes__");
          if (first_is_bytes_like) {
              std::vector<std::string> byte_items; byte_items.reserve(objs.size());
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
              py::gil_scoped_release release;
              return self.sketch(byte_items);
          } else {
              std::vector<uint32_t> int_items; int_items.reserve(objs.size());
              for (const auto& obj : objs) {
                  try {
                      int_items.push_back(py::cast<uint32_t>(obj));
                  } catch (const py::cast_error&) {
                      throw py::value_error("All items must be integers when the first is not string-like.");
                  }
              }
              py::gil_scoped_release release;
              return self.sketch(int_items);
          }
      }, py::arg("items"),
        "Compute sketch for str/bytes or integer lists using FastSimilaritySketchSIMD");

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
      
      // Optimized NumPy array insert method for integers (zero-copy, GIL release)
      .def("insert", [](FastSketchLSH& self, const std::string& key, py::array_t<int32_t> arr) {
          if (arr.size() == 0) {
              throw py::value_error("Array cannot be empty");
          }
          std::vector<int> int_items = numpy_to_vector_zerocopy<int32_t>(arr);
          py::gil_scoped_release release;
          self.insert(key, int_items);
      }, py::arg("key"), py::arg("items"),
          "Insert NumPy int32 array into LSH index (optimized zero-copy)")

      // Optimized list insert method (supports both bytes and strings)
      .def("insert", [](FastSketchLSH& self, const std::string& key, py::list items) {
          if (items.size() == 0) {
              throw py::value_error("List cannot be empty");
          }
          
          // Check if first item is bytes for fast path
          auto first_item = items[0];
          std::vector<std::string> str_items;
          if (py::isinstance<py::bytes>(first_item)) {
              // Fast path for bytes objects
              str_items = bytes_list_to_vector_zerocopy(items);
          } else if (py::isinstance<py::str>(first_item)) {
              // Handle string lists (backward compatibility)
              str_items.reserve(items.size());
              for (auto item : items) {
                  if (py::isinstance<py::str>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else if (py::isinstance<py::bytes>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else {
                      throw py::value_error("All items must be strings or bytes");
                  }
              }
          } else {
              throw py::value_error("Use insert(key, numpy_array) for integers or ensure all items are strings/bytes for this overload");
          }
          py::gil_scoped_release release;
          self.insert(key, str_items);
      }, py::arg("key"), py::arg("items"),
          "Insert list of strings/bytes into LSH index (optimized for bytes)")

      // Optimized NumPy array query method for integers (zero-copy, GIL release)
      .def("query", [](FastSketchLSH& self, py::array_t<int32_t> arr) {
          if (arr.size() == 0) {
              throw py::value_error("Array cannot be empty");
          }
          std::vector<int> int_items = numpy_to_vector_zerocopy<int32_t>(arr);
          py::gil_scoped_release release;
          return self.query(int_items);
      }, py::arg("items"),
          "Query LSH index with NumPy int32 array (optimized zero-copy)")

      // Optimized list query method (supports both bytes and strings)
      .def("query", [](FastSketchLSH& self, py::list items) {
          if (items.size() == 0) {
              throw py::value_error("List cannot be empty");
          }
          
          // Check if first item is bytes for fast path
          auto first_item = items[0];
          std::vector<std::string> str_items;
          if (py::isinstance<py::bytes>(first_item)) {
              // Fast path for bytes objects
              str_items = bytes_list_to_vector_zerocopy(items);
          } else if (py::isinstance<py::str>(first_item)) {
              // Handle string lists (backward compatibility)
              str_items.reserve(items.size());
              for (auto item : items) {
                  if (py::isinstance<py::str>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else if (py::isinstance<py::bytes>(item)) {
                      str_items.push_back(py::cast<std::string>(item));
                  } else {
                      throw py::value_error("All items must be strings or bytes");
                  }
              }
          } else {
              throw py::value_error("Use query(numpy_array) for integers or ensure all items are strings/bytes for this overload");
          }
          py::gil_scoped_release release;
          return self.query(str_items);
      }, py::arg("items"),
          "Query LSH index with list of strings/bytes (optimized for bytes)")

      // Fallback iterable insert method (backward compatibility)
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
              py::gil_scoped_release release;
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
              py::gil_scoped_release release;
              self.insert(key, int_items);
          }
      }, py::arg("key"), py::arg("items"),
          "Insert a set with a key into LSH index\n"
          "Automatically detects whether items are strings or integers")
      
      // Fallback iterable query method (backward compatibility)
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
              py::gil_scoped_release release;
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
              py::gil_scoped_release release;
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

      // Optimized insert with GIL release
      .def("insert",
           [](FastSketchLSHRensa& self, std::size_t key, const FastSimilaritySketch& sketch) {
               py::gil_scoped_release release;
               self.insert(key, sketch);
           },
           py::arg("key"), py::arg("sketch"),
           "Insert a sketch with a numeric key (optimized with GIL release)")

      // Optimized query with GIL release
      .def("query",
           [](const FastSketchLSHRensa& self, const FastSimilaritySketch& sketch) {
               py::gil_scoped_release release;
               return self.query(sketch);
           },
           py::arg("sketch"),
           "Query candidates for a given sketch (optimized with GIL release)")

      // 只读属性
      .def_property_readonly("num_perm",  &FastSketchLSHRensa::num_perm)
      .def_property_readonly("num_bands", &FastSketchLSHRensa::num_bands)
      .def_property_readonly("band_size",&FastSketchLSHRensa::band_size)
      .def_property_readonly("threshold",&FastSketchLSHRensa::threshold);
}