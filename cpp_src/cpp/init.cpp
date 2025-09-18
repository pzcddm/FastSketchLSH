/*
  FastSketchLSH Python↔C++ boundary optimizations (2025-08):

  - NumPy zero-copy fast paths (numeric):
    FastSimilaritySketch::sketch(np.uint32 | np.int32),
    FastSketchLSH::insert/query(np.int32).
    Requirements: 1-D arrays. Reading uses buffer access (no per-element boxing),
    compute runs under GIL release.

  - Bytes fast path (text/bytes-like):
    FastSimilaritySketch::sketch(list[bytes]),
    FastSketchLSH::{insert, query}(list[bytes] | list[str]).
    list[bytes] uses PyBytes_AsStringAndSize to avoid copies.
    list[str] remains supported (back-compat); bytes is fastest.

  - GIL release:
    All compute-heavy code paths (sketch/insert/query, including LSHRensa) release the GIL.

  - Backward compatibility:
    Iterable overloads are preserved. Numeric Python lists and string lists still work, though
    slower than NumPy/bytes fast paths.

  - Guidance:
    Prefer NumPy arrays (np.int32/np.uint32) for numbers and list[bytes] for text
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
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
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

// (buffer-based helper removed)

PYBIND11_MODULE(FastSketchLSH, m) {
    m.attr("__version__") = "0.2.0";
    // Expose OpenMP max threads for diagnostics
    m.def("omp_max_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Return the maximum number of OpenMP threads available (1 if OpenMP disabled)");



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
        "Compute FastSimilaritySketch for NumPy uint32 array (optimized zero-copy)")

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
                  throw py::value_error("FastSimilaritySketch requires non-negative integers");
              }
              int_items.push_back(static_cast<uint32_t>(val));
          }
          py::gil_scoped_release release;
          return self.sketch(int_items);
      }, py::arg("items"),
        "Compute FastSimilaritySketch for NumPy int32 array (optimized zero-copy, converted to uint32)")

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
        "Compute FastSimilaritySketch for list of strings/bytes (optimized for bytes)")

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
        "Compute sketch for str/bytes or integer lists using FastSimilaritySketch")

      // Batch sketch: accept a list of batches. Each batch element can be
      // - NumPy array (np.uint32 or np.int32)
      // - list/tuple/set of ints
      // - list/tuple/set of bytes/str
      // Fast numeric paths return a 2D NumPy array (B, t) to avoid Python int boxing.
      .def("sketch_batch", [](FastSimilaritySketch& self, py::list batches, int num_threads) -> py::object {
           if (batches.size() == 0) {
               throw py::value_error("batches cannot be empty");
           }

           const size_t B = static_cast<size_t>(batches.size());
           const size_t t = static_cast<size_t>(self.t);
           auto first = batches[0];

           // Case 1: list of NumPy arrays (fast path -> returns np.ndarray (B,t))
           if (py::isinstance<py::array>(first)) {
               // uint32 fast path
               if (py::isinstance<py::array_t<uint32_t>>(first)) {
                   // Build pointer arrays to avoid concatenation copy
                   std::unique_ptr<const uint32_t*[]> ptrs(new const uint32_t*[B]);
                   std::unique_ptr<size_t[]> lens(new size_t[B]);
                   for (size_t i = 0; i < B; ++i) {
                       auto arr = py::cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(batches[i]);
                       py::buffer_info bi = arr.request();
                       if (bi.ndim != 1) throw py::value_error("All arrays must be 1D");
                       ptrs[i] = static_cast<const uint32_t*>(bi.ptr);
                       lens[i] = static_cast<size_t>(bi.size);
                   }
                   std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
                   {
                       py::gil_scoped_release release;
                       self.sketch_batch_flat_ptrs(ptrs.get(), lens.get(), B, flat.get(), num_threads);
                   }
                   uint64_t* raw = flat.release();
                   py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
                   return py::array(
                       py::dtype::of<uint64_t>(),
                       std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
                       std::vector<ssize_t>{(ssize_t)(t * sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
                       raw,
                       owner
                   );
               }
               // int32 fast path (validate non-negative, cast to uint32)
               if (py::isinstance<py::array_t<int32_t>>(first)) {
                   size_t total_n = 0;
                   std::vector<size_t> lens; lens.reserve(B);
                   for (size_t i = 0; i < B; ++i) {
                       auto arr = py::cast<py::array_t<int32_t>>(batches[i]);
                       py::buffer_info bi = arr.request();
                       if (bi.ndim != 1) throw py::value_error("All arrays must be 1D");
                       lens.push_back(static_cast<size_t>(bi.size));
                       total_n += static_cast<size_t>(bi.size);
                   }
                   std::unique_ptr<uint32_t[]> data(new uint32_t[total_n]);
                   std::unique_ptr<uint64_t[]> indptr(new uint64_t[B + 1]);
                   size_t pos = 0; indptr[0] = 0;
                   for (size_t i = 0; i < B; ++i) {
                       auto arr = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(batches[i]);
                       py::buffer_info bi = arr.request();
                       const int32_t* src = static_cast<const int32_t*>(bi.ptr);
                       const size_t n = static_cast<size_t>(bi.size);
                       for (size_t j = 0; j < n; ++j) {
                           int32_t v = src[j];
                           if (v < 0) throw py::value_error("FastSimilaritySketch requires non-negative integers");
                           data[pos + j] = static_cast<uint32_t>(v);
                       }
                       pos += n;
                       indptr[i + 1] = static_cast<uint64_t>(pos);
                   }
                   std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
                   {
                       py::gil_scoped_release release;
                       self.sketch_batch_flat_csr(data.get(), indptr.get(), B, flat.get(), num_threads);
                   }
                   uint64_t* raw = flat.release();
                   py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
                   return py::array(
                       py::dtype::of<uint64_t>(),
                       std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
                       std::vector<ssize_t>{(ssize_t)(t * sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
                       raw,
                       owner
                   );
               }
               throw py::value_error("Only int32/uint32 NumPy arrays are supported in batch");
           }

           // Case 2: list/tuple/set of bytes/str or ints
           // Inspect inner container's first element
           auto inner_any = py::reinterpret_borrow<py::object>(batches[0]);
           py::iterable inner_iter;
           try {
               inner_iter = inner_any.cast<py::iterable>();
           } catch (...) {
               throw py::value_error("Each batch element must be an iterable (array/list/tuple/set)");
           }
           if (py::len(inner_iter) == 0) {
               throw py::value_error("Inner iterable cannot be empty");
           }
           auto inner_first = *inner_iter.begin();
           const bool inner_is_bytes_like = py::isinstance<py::bytes>(inner_first)
                                         || py::isinstance<py::str>(inner_first)
                                         || py::hasattr(inner_first, "__bytes__");
           if (inner_is_bytes_like) {
               // Fast pointer path for bytes/str; accept arbitrary bytes (no forced encoding)
               // Flatten all items into arrays of pointers/lengths and indptr per set
               std::vector<uint64_t> indptr; indptr.reserve(B + 1); indptr.push_back(0);
               size_t total_items = 0;
               // First pass: sizes
               for (size_t i = 0; i < B; ++i) {
                   py::object obj = batches[i];
                   PyObject* seq = PySequence_Fast(obj.ptr(), "Each batch element must be a sequence of bytes-like");
                   if (!seq) throw py::error_already_set();
                   const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                   total_items += static_cast<size_t>(n);
                   indptr.push_back(static_cast<uint64_t>(total_items));
                   Py_DECREF(seq);
               }
               std::unique_ptr<const uint8_t*[]> ptrs(new const uint8_t*[total_items]);
               std::unique_ptr<size_t[]> lengths(new size_t[total_items]);
               // To avoid copying for memoryview/buffer objects, retain their Py_buffer until after compute
               std::vector<Py_buffer> retained_buffers; retained_buffers.reserve(total_items);
               size_t pos = 0;
               for (size_t i = 0; i < B; ++i) {
                   py::object obj = batches[i];
                   PyObject* seq = PySequence_Fast(obj.ptr(), "Each batch element must be a sequence of bytes-like");
                   if (!seq) throw py::error_already_set();
                   PyObject** items = PySequence_Fast_ITEMS(seq);
                   const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                   for (Py_ssize_t j = 0; j < n; ++j) {
                       PyObject* it = items[j];
                       // If str, use its UTF-8 view without creating intermediate Python bytes
                       if (PyUnicode_Check(it)) {
                           Py_ssize_t size = 0;
                           const char* s = PyUnicode_AsUTF8AndSize(it, &size);
                           if (!s) { Py_DECREF(seq); throw py::error_already_set(); }
                           ptrs[pos] = reinterpret_cast<const uint8_t*>(s);
                           lengths[pos] = static_cast<size_t>(size);
                       } else if (PyBytes_Check(it)) {
                           char* data = nullptr; Py_ssize_t size = 0;
                           if (PyBytes_AsStringAndSize(it, &data, &size) == -1) { Py_DECREF(seq); throw py::error_already_set(); }
                           ptrs[pos] = reinterpret_cast<const uint8_t*>(data);
                           lengths[pos] = static_cast<size_t>(size);
                       } else if (PyByteArray_Check(it)) {
                           char* data = PyByteArray_AsString(it);
                           Py_ssize_t size = PyByteArray_Size(it);
                           ptrs[pos] = reinterpret_cast<const uint8_t*>(data);
                           lengths[pos] = static_cast<size_t>(size);
                       } else if (PyObject_CheckBuffer(it)) {
                           // Generic buffer protocol (retain view to keep memory alive during compute)
                           Py_buffer view;
                           if (PyObject_GetBuffer(it, &view, PyBUF_SIMPLE) == -1) { Py_DECREF(seq); throw py::error_already_set(); }
                           ptrs[pos] = reinterpret_cast<const uint8_t*>(view.buf);
                           lengths[pos] = static_cast<size_t>(view.len);
                           retained_buffers.push_back(view);
                       } else {
                           Py_DECREF(seq);
                           throw py::value_error("All inner items must be str/bytes/bytearray or buffer");
                       }
                       ++pos;
                   }
                   Py_DECREF(seq);
               }
               std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_bytes(ptrs.get(), lengths.get(), indptr.data(), B, flat.get(), num_threads);
               }
               for (auto& v : retained_buffers) { PyBuffer_Release(&v); }
               uint64_t* raw = flat.release();
               py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
               return py::array(
                   py::dtype::of<uint64_t>(),
                   std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
                   std::vector<ssize_t>{(ssize_t)(t * sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
                   raw,
                   owner
               );
           }

           // Integer iterable fast path: build CSR and return np.ndarray (B,t)
           {
               size_t total_n = 0;
               std::vector<uint64_t> indptr_vec; indptr_vec.reserve(B + 1);
               indptr_vec.push_back(0);
               // First pass: lengths
               for (size_t i = 0; i < B; ++i) {
                   py::object obj = batches[i];
                   PyObject* seq = PySequence_Fast(obj.ptr(), "Each batch element must be a sequence of integers");
                   if (!seq) throw py::error_already_set();
                   const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                   total_n += static_cast<size_t>(n);
                   indptr_vec.push_back(static_cast<uint64_t>(total_n));
                   Py_DECREF(seq);
               }
               std::unique_ptr<uint32_t[]> data(new uint32_t[total_n]);
               std::unique_ptr<uint64_t[]> indptr(new uint64_t[B + 1]);
               for (size_t i = 0; i < B + 1; ++i) indptr[i] = indptr_vec[i];
               // Second pass: fill data with minimal overhead
               size_t pos = 0;
               for (size_t i = 0; i < B; ++i) {
                   py::object obj = batches[i];
                   PyObject* seq = PySequence_Fast(obj.ptr(), "Each batch element must be a sequence of integers");
                   if (!seq) throw py::error_already_set();
                   PyObject** items = PySequence_Fast_ITEMS(seq);
                   const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                   for (Py_ssize_t j = 0; j < n; ++j) {
                       unsigned long long v = PyLong_AsUnsignedLongLong(items[j]);
                       if (v == (unsigned long long)-1 && PyErr_Occurred()) {
                           Py_DECREF(seq);
                           throw py::value_error("All inner items must be integers");
                       }
                       if (v > 0xFFFFFFFFull) {
                           Py_DECREF(seq);
                           throw py::value_error("Integer exceeds uint32 range");
                       }
                       data[pos + static_cast<size_t>(j)] = static_cast<uint32_t>(v);
                   }
                   pos += static_cast<size_t>(n);
                   Py_DECREF(seq);
               }
               std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_csr(data.get(), indptr.get(), B, flat.get(), num_threads);
               }
               uint64_t* raw = flat.release();
               py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
               return py::array(
                   py::dtype::of<uint64_t>(),
                   std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
                   std::vector<ssize_t>{(ssize_t)(t * sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
                   raw,
                   owner
               );
           }
       }, py::arg("batches"), py::arg("num_threads") = 0,
          "Compute sketches for a batch.\n"
          "batches: list of (np.int32/np.uint32 arrays) or list/tuple/set of ints or bytes/str.\n"
          "num_threads: 0 uses all threads (if OpenMP enabled). 1 forces single-thread.")

      // Removed sketch_batch_flat(list-based). Use sketch_batch_flat_csr for flat outputs.

      // CSR zero-copy numeric batch: (data: np.uint32, indptr: np.uint64) -> np.ndarray (B, t)
      .def("sketch_batch_flat_csr", [](FastSimilaritySketch& self,
                                        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> data,
                                        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indptr,
                                        int num_threads) {
           py::buffer_info bd = data.request();
           py::buffer_info bi = indptr.request();
           if (bi.ndim != 1 || bd.ndim != 1) throw py::value_error("data and indptr must be 1D arrays");
           if (bi.size < 2) throw py::value_error("indptr must have length >= 2");
           const size_t B = static_cast<size_t>(bi.size - 1);
           const size_t t = static_cast<size_t>(self.t);
           uint32_t* dptr = static_cast<uint32_t*>(bd.ptr);
           uint64_t* iptr = static_cast<uint64_t*>(bi.ptr);
           // Allocate flat output and compute under GIL release
           std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
           {
               py::gil_scoped_release release;
               self.sketch_batch_flat_csr(dptr, iptr, B, flat.get(), num_threads);
           }
           // Wrap as NumPy array without copy
           uint64_t* raw = flat.release();
           py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
           return py::array(
               py::dtype::of<uint64_t>(),
               std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
               std::vector<ssize_t>{(ssize_t)(t*sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
               raw,
               owner
           );
      }, py::arg("data"), py::arg("indptr"), py::arg("num_threads") = 0,
         "CSR zero-copy batch: data(np.uint32), indptr(np.uint64 length B+1) -> np.ndarray (B,t)")
 ;

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