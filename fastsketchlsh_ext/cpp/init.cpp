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
#include "../include/LSH.h"

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

      .def("sketch_utf8_fast", [](FastSimilaritySketch& self, py::list items) {
          if (items.size() == 0) {
              throw py::value_error("List cannot be empty");
          }
          const Py_ssize_t n = static_cast<Py_ssize_t>(items.size());
          std::vector<const uint8_t*> ptrs(static_cast<size_t>(n));
          std::vector<size_t> lengths(static_cast<size_t>(n));
          std::vector<py::bytes> utf8_cache;
          utf8_cache.reserve(static_cast<size_t>(n));
          for (Py_ssize_t i = 0; i < n; ++i) {
              py::handle item = items[i];
              PyObject* obj = item.ptr();
              if (!PyUnicode_Check(obj)) {
                  throw py::value_error("sketch_utf8_fast expects all items to be str");
              }
              if (PyUnicode_READY(obj) == -1) {
                  throw py::error_already_set();
              }
              const size_t idx = static_cast<size_t>(i);
              if (PyUnicode_IS_ASCII(obj)) {
                  Py_ssize_t size = PyUnicode_GET_LENGTH(obj);
                  ptrs[idx] = reinterpret_cast<const uint8_t*>(PyUnicode_1BYTE_DATA(obj));
                  lengths[idx] = static_cast<size_t>(size);
              } else {
                  PyObject* utf8_obj = PyUnicode_AsUTF8String(obj);
                  if (!utf8_obj) {
                      throw py::error_already_set();
                  }
                  utf8_cache.emplace_back(py::reinterpret_steal<py::bytes>(utf8_obj));
                  Py_ssize_t size = PyBytes_GET_SIZE(utf8_cache.back().ptr());
                  ptrs[idx] = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(utf8_cache.back().ptr()));
                  lengths[idx] = static_cast<size_t>(size);
              }
          }
          py::gil_scoped_release release;
          return self.sketch_utf8_views(ptrs.data(), lengths.data(), static_cast<size_t>(n));
      }, py::arg("items"),
        "Experimental fast path that sketches list[str] via zero-copy UTF-8 views (ASCII strings stay zero-copy).")

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

    

    // ===================== New band-parallel LSH bindings =====================
    py::enum_<LSH::BandHashKind>(m, "BandHashKind")
        .value("splitmix64", LSH::BandHashKind::splitmix64)
        .value("wyhash_final", LSH::BandHashKind::wyhash_final)
        .export_values();

    py::class_<LSH>(m, "LSH")
      .def(py::init<std::size_t, std::size_t, LSH::BandHashKind, std::uint64_t, int>(),
           py::arg("num_perm"),
           py::arg("num_bands"),
           py::arg("hash_kind") = LSH::BandHashKind::splitmix64,
           py::arg("seed") = 0x9e3779b97f4a7c15ULL,
           py::arg("num_threads") = 0,
           "Initialize band-parallel LSH (num_threads<=0 uses OpenMP default)")
      .def_property_readonly("num_threads", &LSH::num_threads,
           "Configured OpenMP thread count (0 means auto)")
      .def("set_num_threads", &LSH::set_num_threads, py::arg("num_threads"),
           "Update the OpenMP thread count (<=0 means auto, requires OpenMP for >1)")
      .def("reserve", &LSH::reserve, py::arg("expected_num_items"),
           "Reserve internal capacity for expected number of items")
      .def("clear", &LSH::clear, "Clear all tables and reset state")

      // Build from 2D NumPy ndarray (B, t), dtype=uint64, contiguous or strided
      .def("build_from_batch", [](LSH& self,
                                   py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> arr) {
           py::buffer_info bi = arr.request();
           if (bi.ndim != 2) {
               throw py::value_error("Input must be a 2D array of shape (B, t)");
           }
           const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
           const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
           if (t != self.num_perm()) {
               throw py::value_error("t must equal num_perm");
           }
           const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);
           {
               py::gil_scoped_release release;
               self.build_from_batch(base, B, t);
           }
       }, py::arg("ndarray"),
          "Build from 2D NumPy ndarray (uint64) with zero/low-copy")

      // Build from list of 1D NumPy arrays (each length t, dtype=uint64)
      .def("build_from_batch", [](LSH& self, py::list rows) {
           const std::size_t B = static_cast<std::size_t>(rows.size());
           if (B == 0) return;
           const std::size_t t = self.num_perm();
           std::unique_ptr<const std::uint64_t*[]> ptrs(new const std::uint64_t*[B]);
           for (std::size_t i = 0; i < B; ++i) {
               auto arr = py::cast<py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>>(rows[i]);
               py::buffer_info bi = arr.request();
               if (bi.ndim != 1) throw py::value_error("Each array must be 1D");
               if (static_cast<std::size_t>(bi.size) != t) throw py::value_error("Row length must equal num_perm");
               ptrs[i] = static_cast<const std::uint64_t*>(bi.ptr);
           }
           {
               py::gil_scoped_release release;
               self.build_from_batch(ptrs.get(), B, t);
           }
       }, py::arg("rows"),
          "Build from list of NumPy arrays (uint64, length t) without copies")

      // Build from list of Python lists (will copy into a temporary (B,t) buffer)
      .def("build_from_batch", [](LSH& self, py::object py_rows) {
           PyObject* seq = PySequence_Fast(py_rows.ptr(), "rows must be a sequence");
           if (!seq) throw py::error_already_set();
           const Py_ssize_t Bp = PySequence_Fast_GET_SIZE(seq);
           const std::size_t B = static_cast<std::size_t>(Bp);
           const std::size_t t = self.num_perm();
           std::vector<std::uint64_t> buf;
           buf.resize(B * t);
           for (Py_ssize_t i = 0; i < Bp; ++i) {
               PyObject* row_obj = PySequence_Fast_GET_ITEM(seq, i);
               PyObject* row_seq = PySequence_Fast(row_obj, "Each row must be a sequence");
               if (!row_seq) { Py_DECREF(seq); throw py::error_already_set(); }
               const Py_ssize_t n = PySequence_Fast_GET_SIZE(row_seq);
               if (static_cast<std::size_t>(n) != t) { Py_DECREF(row_seq); Py_DECREF(seq); throw py::value_error("Row length must equal num_perm"); }
               PyObject** items = PySequence_Fast_ITEMS(row_seq);
               std::uint64_t* out = buf.data() + static_cast<std::size_t>(i) * t;
               for (Py_ssize_t j = 0; j < n; ++j) {
                   unsigned long long v = PyLong_AsUnsignedLongLong(items[j]);
                   if (v == (unsigned long long)-1 && PyErr_Occurred()) { Py_DECREF(row_seq); Py_DECREF(seq); throw py::value_error("All items must be integers"); }
                   out[static_cast<std::size_t>(j)] = static_cast<std::uint64_t>(v);
               }
               Py_DECREF(row_seq);
           }
           Py_DECREF(seq);
           const std::uint64_t* base = buf.data();
           {
               py::gil_scoped_release release;
               self.build_from_batch(base, B, t);
           }
       }, py::arg("rows"),
          "Build from list of Python lists (copied once into a temporary buffer)")

      // Query candidates: 1D NumPy array (t,) uint64 → Python list[int]
      .def("query_candidates", [](const LSH& self,
                                   py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> digest) {
           py::buffer_info bi = digest.request();
           const std::size_t t = static_cast<std::size_t>(bi.size);
           const std::uint64_t* ptr = static_cast<const std::uint64_t*>(bi.ptr);
           py::gil_scoped_release release;
           return self.query_candidates(ptr, t);
       }, py::arg("digest"), "Query candidates for a single digest (NumPy uint64 array)")

      // Query candidates: Python list of ints → Python list[int]
      .def("query_candidates", [](const LSH& self, py::iterable py_digest) {
           std::vector<std::uint64_t> buf;
           buf.reserve(self.num_perm());
           for (auto item : py_digest) {
               unsigned long long v = PyLong_AsUnsignedLongLong(item.ptr());
               if (v == (unsigned long long)-1 && PyErr_Occurred()) throw py::value_error("All items must be integers");
               buf.push_back(static_cast<std::uint64_t>(v));
           }
           py::gil_scoped_release release;
           return self.query_candidates(buf.data(), buf.size());
       }, py::arg("digest"), "Query candidates for a single digest (Python list of ints)")

      // Batch query: return CSR-style (flat candidates uint64, indptr uint64) for ndarray (B,t)
      .def("batch_query_csr", [](const LSH& self,
                                            py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> arr) {
           py::buffer_info bi = arr.request();
           if (bi.ndim != 2) throw py::value_error("Input must be 2D (B,t) uint64 array");
           const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
           const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
           const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);
           std::vector<std::size_t> flat;
           std::vector<std::uint64_t> indptr;
           {
               py::gil_scoped_release release;
               self.query_candidates_batch(base, B, t, flat, indptr);
           }
           // Wrap flat as uint64 array
           const ssize_t nflat = static_cast<ssize_t>(flat.size());
           std::uint64_t* flat_raw = new std::uint64_t[static_cast<std::size_t>(nflat)];
           for (ssize_t i = 0; i < nflat; ++i) flat_raw[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(flat[static_cast<std::size_t>(i)]);
           py::capsule owner_flat(flat_raw, [](void* f){ delete[] reinterpret_cast<std::uint64_t*>(f); });
           py::array flat_arr(
               py::dtype::of<std::uint64_t>(),
               std::vector<ssize_t>{nflat},
               std::vector<ssize_t>{static_cast<ssize_t>(sizeof(std::uint64_t))},
               flat_raw,
               owner_flat
           );
           // Wrap indptr as uint64 array
           const ssize_t nip = static_cast<ssize_t>(indptr.size());
           std::uint64_t* ip_raw = new std::uint64_t[static_cast<std::size_t>(nip)];
           for (ssize_t i = 0; i < nip; ++i) ip_raw[static_cast<std::size_t>(i)] = indptr[static_cast<std::size_t>(i)];
           py::capsule owner_ip(ip_raw, [](void* f){ delete[] reinterpret_cast<std::uint64_t*>(f); });
           py::array indptr_arr(
               py::dtype::of<std::uint64_t>(),
               std::vector<ssize_t>{nip},
               std::vector<ssize_t>{static_cast<ssize_t>(sizeof(std::uint64_t))},
               ip_raw,
               owner_ip
           );
           return py::make_tuple(flat_arr, indptr_arr);
       }, py::arg("ndarray"),
          "Batch query returning (flat uint64 array, indptr uint64 array)")

      // Batch query: return Python list-of-lists of candidates for ndarray (B,t)
      .def("batch_query", [](const LSH& self,
                                              py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> arr) {
           py::buffer_info bi = arr.request();
           if (bi.ndim != 2) throw py::value_error("Input must be 2D (B,t) uint64 array");
           const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
           const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
           const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);
           std::vector<std::size_t> flat;
           std::vector<std::uint64_t> indptr;
           {
               py::gil_scoped_release release;
               self.query_candidates_batch(base, B, t, flat, indptr);
           }
           py::list outer(static_cast<py::ssize_t>(B));
           for (std::size_t i = 0; i < B; ++i) {
               const std::size_t start = static_cast<std::size_t>(indptr[i]);
               const std::size_t end = static_cast<std::size_t>(indptr[i+1]);
               const std::size_t len = end - start;
               py::list inner(static_cast<py::ssize_t>(len));
               for (std::size_t j = 0; j < len; ++j) {
                   const std::size_t id = flat[start + j];
                   PyObject* pyint = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(id));
                   // Steals reference
                   PyList_SET_ITEM(inner.ptr(), static_cast<Py_ssize_t>(j), pyint);
               }
               // Steal reference for inner into outer
               PyList_SET_ITEM(outer.ptr(), static_cast<Py_ssize_t>(i), inner.release().ptr());
           }
           return outer;
       }, py::arg("ndarray"),
          "Batch query returning Python list-of-lists (minimized allocations)")

      // Read-only properties
      .def_property_readonly("num_perm",  &LSH::num_perm)
      .def_property_readonly("num_bands", &LSH::num_bands)
      .def_property_readonly("band_size", &LSH::band_size)
      // threshold removed
    ;
}
