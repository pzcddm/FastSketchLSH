/*
  FastSketchLSH Python↔C++ boundary – consolidated API (v1.0.0)

  Public surface (9 methods):
    FastSimilaritySketch(size=128, seed=42)
      __call__(items, prehashed=False)
      batch(rows, prehashed=False, num_threads=0)
      batch_csr(data, indptr, prehashed=False, num_threads=0)
    LSH(num_perm, num_bands, seed=..., num_threads=0)
      insert(data)          – 3 overloads (2D ndarray, list[ndarray], list[list])
      query(input, format=None)  – 1D→single, 2D→batch, format="csr"→CSR
      duplicates(arr, self_start=0)
      + reserve / clear / set_num_threads / read-only properties

  All compute-heavy paths release the GIL.
  Fast paths: NumPy zero-copy, bytes zero-copy, ASCII zero-copy UTF-8 views,
  prehashed CSR, fused str-list chunked hash+sketch.
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <unicodeobject.h>
#include <cstddef>  // For size_t and ssize_t
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// NumPy C API for direct array access (bypass buffer protocol)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL FASTSKETCHLSH_ARRAY_API
#include <numpy/arrayobject.h>

#include "../include/fastsketch.h"
#include "../include/LSH.h"

namespace py = pybind11;

// ===================== Optimized Helper Functions =====================

// Fast path for bytes objects - zero copy access
inline std::vector<std::string> bytes_list_to_vector_zerocopy(py::list items) {
    std::vector<std::string> result;
    result.reserve(items.size());

    for (auto item : items) {
        if (py::isinstance<py::bytes>(item)) {
            // Zero-copy access to bytes data
            char* data = nullptr;
            Py_ssize_t size = 0;
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

// Helper: return a (B, t) uint64 NumPy array that owns the flat buffer
static inline py::array wrap_flat_as_2d(uint64_t* raw, size_t B, size_t t) {
    py::capsule owner(raw, [](void* f){ delete[] reinterpret_cast<uint64_t*>(f); });
    return py::array(
        py::dtype::of<uint64_t>(),
        std::vector<ssize_t>{(ssize_t)B, (ssize_t)t},
        std::vector<ssize_t>{(ssize_t)(t * sizeof(uint64_t)), (ssize_t)sizeof(uint64_t)},
        raw, owner
    );
}

PYBIND11_MODULE(FastSketchLSH, m) {
    // Initialize NumPy C API (required for direct array access)
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }

    m.attr("__version__") = "1.0.0";
    // Expose OpenMP max threads for diagnostics
    m.def("omp_max_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Return the maximum number of OpenMP threads available (1 if OpenMP disabled)");


    // ===================== FastSimilaritySketch =====================

    py::class_<FastSimilaritySketch>(m, "FastSimilaritySketch")
      .def( py::init<size_t, uint64_t>(),
            py::arg("size") = 128,
            py::arg("seed") = 42,
            "Initialize FastSimilaritySketch with:\n"
            "  size: Number of sketch dimensions\n"
            "  seed: Random seed (default=42)")

      // ── __call__(items, prehashed=False) ──────────────────────────────
      .def("__call__", [](FastSimilaritySketch& self, py::object items, bool prehashed) -> std::vector<uint64_t> {
          PyObject* obj_ptr = items.ptr();

          if (prehashed) {
              // ═══ PREHASHED PATH ═══
              if (PyArray_Check(obj_ptr)) {
                  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj_ptr);
                  if (PyArray_NDIM(arr) != 1)
                      throw py::value_error("NumPy array must be 1-dimensional");
                  if (PyArray_SIZE(arr) == 0)
                      throw py::value_error("Array cannot be empty");
                  int dtype = PyArray_TYPE(arr);
                  if (dtype == NPY_UINT64) {
                      auto typed = py::cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(items);
                      py::buffer_info buf = typed.request();
                      const auto* ptr = static_cast<const uint64_t*>(buf.ptr);
                      const size_t n = static_cast<size_t>(buf.size);
                      py::gil_scoped_release release;
                      return self.sketch_prehashed(ptr, n);
                  } else if (dtype == NPY_INT64) {
                      // Reinterpret int64 bit patterns as uint64 (zero-copy)
                      auto typed = py::cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(items);
                      py::buffer_info buf = typed.request();
                      const auto* ptr = reinterpret_cast<const uint64_t*>(static_cast<const int64_t*>(buf.ptr));
                      const size_t n = static_cast<size_t>(buf.size);
                      py::gil_scoped_release release;
                      return self.sketch_prehashed(ptr, n);
                  } else {
                      // Fallback: forcecast other numeric dtypes to uint64
                      auto typed = py::cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(items);
                      py::buffer_info buf = typed.request();
                      const auto* ptr = static_cast<const uint64_t*>(buf.ptr);
                      const size_t n = static_cast<size_t>(buf.size);
                      py::gil_scoped_release release;
                      return self.sketch_prehashed(ptr, n);
                  }
              }
              // list/tuple fallback for prehashed
              PyObject* seq = PySequence_Fast(obj_ptr, "items must be a list or tuple of integers");
              if (!seq) throw py::error_already_set();
              const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
              if (n == 0) {
                  Py_DECREF(seq);
                  throw py::value_error("Items cannot be empty");
              }
              std::unique_ptr<uint64_t[]> buf(new uint64_t[static_cast<size_t>(n)]);
              PyObject** seq_items = PySequence_Fast_ITEMS(seq);
              for (Py_ssize_t i = 0; i < n; ++i) {
                  unsigned long long v = PyLong_AsUnsignedLongLong(seq_items[i]);
                  if (v == (unsigned long long)-1 && PyErr_Occurred()) {
                      Py_DECREF(seq);
                      throw py::value_error("All items must be non-negative integers fitting in uint64");
                  }
                  buf[static_cast<size_t>(i)] = static_cast<uint64_t>(v);
              }
              Py_DECREF(seq);
              const uint64_t* ptr = buf.get();
              py::gil_scoped_release release;
              return self.sketch_prehashed(ptr, static_cast<size_t>(n));
          }

          // ═══ NON-PREHASHED PATH ═══

          // --- NumPy typed arrays ---
          if (PyArray_Check(obj_ptr)) {
              PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj_ptr);
              if (PyArray_NDIM(arr) != 1)
                  throw py::value_error("NumPy array must be 1-dimensional");
              if (PyArray_SIZE(arr) == 0)
                  throw py::value_error("Array cannot be empty");

              int dtype = PyArray_TYPE(arr);
              if (dtype == NPY_UINT32) {
                  // uint32 zero-copy fast path
                  auto typed = py::cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(items);
                  py::buffer_info buf = typed.request();
                  const auto* ptr = static_cast<const uint32_t*>(buf.ptr);
                  const size_t n = static_cast<size_t>(buf.size);
                  py::gil_scoped_release release;
                  return self.sketch(ptr, n);
              } else if (dtype == NPY_INT32) {
                  // int32: validate non-negative, convert to uint32
                  auto typed = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(items);
                  py::buffer_info buf = typed.request();
                  const auto* src = static_cast<const int32_t*>(buf.ptr);
                  const size_t n = static_cast<size_t>(buf.size);
                  std::vector<uint32_t> int_items;
                  int_items.reserve(n);
                  for (size_t i = 0; i < n; ++i) {
                      int32_t val = src[i];
                      if (val < 0)
                          throw py::value_error("FastSimilaritySketch requires non-negative integers");
                      int_items.push_back(static_cast<uint32_t>(val));
                  }
                  py::gil_scoped_release release;
                  return self.sketch(int_items);
              } else if (dtype == NPY_OBJECT) {
                  // NumPy object array of strings
                  PyObject** data = reinterpret_cast<PyObject**>(PyArray_DATA(arr));
                  const Py_ssize_t n = static_cast<Py_ssize_t>(PyArray_SIZE(arr));
                  if (!PyUnicode_Check(data[0]))
                      throw py::value_error("NumPy object array must contain strings");

                  std::vector<const uint8_t*> ptrs(static_cast<size_t>(n));
                  std::vector<size_t> lengths(static_cast<size_t>(n));
                  std::vector<py::bytes> utf8_cache;
                  utf8_cache.reserve(static_cast<size_t>(n));

                  for (Py_ssize_t i = 0; i < n; ++i) {
                      PyObject* str_obj = data[i];
                      if (PyUnicode_READY(str_obj) == -1)
                          throw py::error_already_set();
                      const size_t idx = static_cast<size_t>(i);
                      if (PyUnicode_IS_ASCII(str_obj)) {
                          Py_ssize_t str_size = PyUnicode_GET_LENGTH(str_obj);
                          ptrs[idx] = reinterpret_cast<const uint8_t*>(PyUnicode_1BYTE_DATA(str_obj));
                          lengths[idx] = static_cast<size_t>(str_size);
                      } else {
                          PyObject* utf8_obj = PyUnicode_AsUTF8String(str_obj);
                          if (!utf8_obj) throw py::error_already_set();
                          utf8_cache.emplace_back(py::reinterpret_steal<py::bytes>(utf8_obj));
                          Py_ssize_t str_size = PyBytes_GET_SIZE(utf8_cache.back().ptr());
                          ptrs[idx] = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(utf8_cache.back().ptr()));
                          lengths[idx] = static_cast<size_t>(str_size);
                      }
                  }
                  py::gil_scoped_release release;
                  return self.sketch_utf8_views(ptrs.data(), lengths.data(), static_cast<size_t>(n));
              } else {
                  throw py::value_error("For numeric arrays, use np.int32 or np.uint32. For strings, use dtype=object.");
              }
          }

          // --- Python tuple of ints ---
          if (PyTuple_Check(obj_ptr)) {
              const Py_ssize_t n = PyTuple_GET_SIZE(obj_ptr);
              if (n == 0) throw py::value_error("Tuple cannot be empty");
              PyObject* first = PyTuple_GET_ITEM(obj_ptr, 0);
              if (!PyLong_Check(first))
                  throw py::value_error("For tuples, only integer elements are supported in fast path");

              std::unique_ptr<uint32_t[]> int_items(new uint32_t[static_cast<size_t>(n)]);
              for (Py_ssize_t i = 0; i < n; ++i) {
                  PyObject* item = PyTuple_GET_ITEM(obj_ptr, i);
                  long value = PyLong_AsLong(item);
                  int_items[static_cast<size_t>(i)] = static_cast<uint32_t>(value);
              }
              if (PyErr_Occurred()) {
                  PyErr_Clear();
                  throw py::value_error("All items must be non-negative integers fitting in uint32");
              }
              const uint32_t* ptr = int_items.get();
              py::gil_scoped_release release;
              return self.sketch(ptr, static_cast<size_t>(n));
          }

          // --- Python list: dispatch by first element type ---
          if (PyList_Check(obj_ptr)) {
              if (PyList_GET_SIZE(obj_ptr) == 0)
                  throw py::value_error("List cannot be empty");

              PyObject* first_item = PyList_GET_ITEM(obj_ptr, 0);

              // strings: zero-copy UTF-8 views
              if (PyUnicode_Check(first_item)) {
                  const Py_ssize_t n = PyList_GET_SIZE(obj_ptr);
                  std::vector<const uint8_t*> ptrs(static_cast<size_t>(n));
                  std::vector<size_t> lengths(static_cast<size_t>(n));
                  std::vector<py::bytes> utf8_cache;
                  utf8_cache.reserve(static_cast<size_t>(n));

                  for (Py_ssize_t i = 0; i < n; ++i) {
                      PyObject* str_obj = PyList_GET_ITEM(obj_ptr, i);
                      if (PyUnicode_READY(str_obj) == -1)
                          throw py::error_already_set();
                      const size_t idx = static_cast<size_t>(i);
                      if (PyUnicode_IS_ASCII(str_obj)) {
                          Py_ssize_t size = PyUnicode_GET_LENGTH(str_obj);
                          ptrs[idx] = reinterpret_cast<const uint8_t*>(PyUnicode_1BYTE_DATA(str_obj));
                          lengths[idx] = static_cast<size_t>(size);
                      } else {
                          PyObject* utf8_obj = PyUnicode_AsUTF8String(str_obj);
                          if (!utf8_obj) throw py::error_already_set();
                          utf8_cache.emplace_back(py::reinterpret_steal<py::bytes>(utf8_obj));
                          Py_ssize_t size = PyBytes_GET_SIZE(utf8_cache.back().ptr());
                          ptrs[idx] = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(utf8_cache.back().ptr()));
                          lengths[idx] = static_cast<size_t>(size);
                      }
                  }
                  py::gil_scoped_release release;
                  return self.sketch_utf8_views(ptrs.data(), lengths.data(), static_cast<size_t>(n));
              }

              // bytes: zero-copy
              if (PyBytes_Check(first_item)) {
                  py::list items_list = py::reinterpret_borrow<py::list>(items);
                  std::vector<std::string> byte_items = bytes_list_to_vector_zerocopy(items_list);
                  py::gil_scoped_release release;
                  return self.sketch(byte_items);
              }

              // ints
              if (PyLong_Check(first_item)) {
                  const Py_ssize_t n = PyList_GET_SIZE(obj_ptr);
                  std::unique_ptr<uint32_t[]> int_items(new uint32_t[static_cast<size_t>(n)]);
                  for (Py_ssize_t i = 0; i < n; ++i) {
                      PyObject* item = PyList_GET_ITEM(obj_ptr, i);
                      long value = PyLong_AsLong(item);
                      int_items[static_cast<size_t>(i)] = static_cast<uint32_t>(value);
                  }
                  if (PyErr_Occurred()) {
                      PyErr_Clear();
                      throw py::value_error("All items must be non-negative integers fitting in uint32");
                  }
                  const uint32_t* ptr = int_items.get();
                  py::gil_scoped_release release;
                  return self.sketch(ptr, static_cast<size_t>(n));
              }

              throw py::value_error("List items must be strings, bytes, or integers");
          }

          throw py::value_error("Input must be a numpy array, list, or tuple");
      }, py::arg("items"), py::arg("prehashed") = false,
        "Compute sketch for items.\n"
        "  prehashed=False: items are tokens (np.uint32/int32, list[str/bytes/int], tuple[int])\n"
        "  prehashed=True:  items are pre-hashed uint64 values (np.uint64/int64, list[int])")

      // ── batch(batches, prehashed=False, num_threads=0) ────────────────
      .def("batch", [](FastSimilaritySketch& self, py::list batches, bool prehashed, int num_threads) -> py::object {
           if (batches.size() == 0) {
               throw py::value_error("batches cannot be empty");
           }
           const size_t B = static_cast<size_t>(batches.size());
           const size_t t = static_cast<size_t>(self.t);

           if (prehashed) {
               // ═══ PREHASHED BATCH ═══
               std::unique_ptr<const uint64_t*[]> ptrs(new const uint64_t*[B]);
               std::unique_ptr<size_t[]> lens(new size_t[B]);
               std::vector<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>> handles(B);

               for (size_t i = 0; i < B; ++i) {
                   handles[i] = py::cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(batches[i]);
                   py::buffer_info bi = handles[i].request();
                   if (bi.ndim != 1) throw py::value_error("All arrays must be 1D");
                   ptrs[i] = static_cast<const uint64_t*>(bi.ptr);
                   lens[i] = static_cast<size_t>(bi.size);
               }

               std::unique_ptr<uint64_t[]> indptr(new uint64_t[B + 1]);
               indptr[0] = 0;
               size_t total = 0;
               for (size_t i = 0; i < B; ++i) {
                   total += lens[i];
                   indptr[i + 1] = static_cast<uint64_t>(total);
               }

               std::unique_ptr<uint64_t[]> flat_data(new uint64_t[total]);
               for (size_t i = 0; i < B; ++i) {
                   std::memcpy(flat_data.get() + indptr[i], ptrs[i], lens[i] * sizeof(uint64_t));
               }

               std::unique_ptr<uint64_t[]> flat_out(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_csr_prehashed(flat_data.get(), indptr.get(), B, flat_out.get(), num_threads);
               }
               return wrap_flat_as_2d(flat_out.release(), B, t);
           }

           // ═══ NON-PREHASHED BATCH ═══
           auto first = batches[0];

           // Case 1: list of NumPy arrays (fast path -> returns np.ndarray (B,t))
           if (py::isinstance<py::array>(first)) {
               // uint32 fast path
               if (py::isinstance<py::array_t<uint32_t>>(first)) {
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
                   return wrap_flat_as_2d(flat.release(), B, t);
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
                   return wrap_flat_as_2d(flat.release(), B, t);
               }
               throw py::value_error("Only int32/uint32 NumPy arrays are supported in batch");
           }

           // Case 2: list/tuple/set of bytes/str or ints
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
               // SINGLE-PASS OPTIMIZED PATH: detect type and process in one pass
               PyObject* first_seq = PySequence_Fast(batches[0].ptr(), "");
               if (!first_seq) throw py::error_already_set();
               PyObject* first_item = PySequence_Fast_ITEMS(first_seq)[0];
               const bool is_str_data = PyUnicode_CheckExact(first_item);
               const bool is_bytes_data = PyBytes_CheckExact(first_item);
               Py_DECREF(first_seq);

               // ── FAST PATH: list[list[str]] chunked fused hash+sketch ────────
               // Single-thread only: fuse hashing under GIL with sketch kernel per chunk.
               // Multi-thread: fall through to the ptrs/lengths path so that
               // sketch_batch_flat_bytes can parallelize BOTH hashing and sketching.
               if (is_str_data && (num_threads == 1)) {
                   bool all_lists = true;
                   for (size_t _ci = 0; _ci < B && all_lists; ++_ci)
                       all_lists = PyList_Check(batches[_ci].ptr());
                   if (all_lists) {
                       static const size_t CHUNK = 128;
                       std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
                       std::vector<uint64_t> chunk_hashes;
                       chunk_hashes.reserve(CHUNK * 512);
                       std::vector<uint64_t> chunk_indptr(CHUNK + 1);
                       bool had_error = false;
                       for (size_t cs = 0; cs < B && !had_error; cs += CHUNK) {
                           const size_t ce  = std::min(cs + CHUNK, B);
                           const size_t csz = ce - cs;
                           chunk_indptr[0] = 0;
                           size_t ctotal = 0;
                           for (size_t i = 0; i < csz; ++i) {
                               ctotal += static_cast<size_t>(
                                   PyList_GET_SIZE(batches[cs + i].ptr()));
                               chunk_indptr[i + 1] = static_cast<uint64_t>(ctotal);
                           }
                           if (chunk_hashes.size() < ctotal)
                               chunk_hashes.resize(ctotal);
                           size_t pos = 0;
                           for (size_t i = 0; i < csz && !had_error; ++i) {
                               PyObject* inner = batches[cs + i].ptr();
                               const Py_ssize_t n = PyList_GET_SIZE(inner);
                               for (Py_ssize_t j = 0; j < n; ++j) {
                                   PyObject* str_obj = PyList_GET_ITEM(inner, j);
                                   if (PyUnicode_IS_ASCII(str_obj)) {
                                       const uint8_t* p = reinterpret_cast<const uint8_t*>(
                                           PyUnicode_1BYTE_DATA(str_obj));
                                       const size_t l = static_cast<size_t>(
                                           PyUnicode_GET_LENGTH(str_obj));
#if defined(FASTSKETCH_USE_FNV1A)
                                       chunk_hashes[pos++] = fnv1a64(p, l);
#else
                                       chunk_hashes[pos++] = fxhash64(p, l);
#endif
                                   } else {
                                       Py_ssize_t sz = 0;
                                       const char* s = PyUnicode_AsUTF8AndSize(str_obj, &sz);
                                       if (!s) { had_error = true; break; }
#if defined(FASTSKETCH_USE_FNV1A)
                                       chunk_hashes[pos++] = fnv1a64(
                                           reinterpret_cast<const uint8_t*>(s),
                                           static_cast<size_t>(sz));
#else
                                       chunk_hashes[pos++] = fxhash64(
                                           reinterpret_cast<const uint8_t*>(s),
                                           static_cast<size_t>(sz));
#endif
                                   }
                               }
                           }
                           if (had_error) break;
                           {
                               py::gil_scoped_release release;
                               uint64_t* chunk_out = flat.get() + cs * t;
                               self.sketch_batch_flat_csr_prehashed(
                                   chunk_hashes.data(), chunk_indptr.data(),
                                   csz, chunk_out, num_threads);
                           }
                       }
                       if (had_error) throw py::error_already_set();
                       return wrap_flat_as_2d(flat.release(), B, t);
                   }
                   // else: fall through — handles tuples/sets via PySequence_Fast
               }

               // Single pass: keep sequences alive, count items, and process
               std::vector<PyObject*> sequences; sequences.reserve(B);
               std::vector<uint64_t> indptr; indptr.reserve(B + 1); indptr.push_back(0);
               size_t total_items = 0;

               for (size_t i = 0; i < B; ++i) {
                   PyObject* seq = PySequence_Fast(batches[i].ptr(), "Each batch element must be a sequence");
                   if (!seq) {
                       for (auto* s : sequences) Py_DECREF(s);
                       throw py::error_already_set();
                   }
                   sequences.push_back(seq);
                   const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                   total_items += static_cast<size_t>(n);
                   indptr.push_back(static_cast<uint64_t>(total_items));
               }

               std::unique_ptr<const uint8_t*[]> ptrs(new const uint8_t*[total_items]);
               std::unique_ptr<size_t[]> lengths(new size_t[total_items]);
               std::vector<Py_buffer> retained_buffers; retained_buffers.reserve(total_items / 10);
               std::vector<py::bytes> utf8_cache; utf8_cache.reserve(total_items / 10);

               size_t pos = 0;

               if (is_str_data) {
                   bool had_error = false;
                   for (size_t i = 0; i < B && !had_error; ++i) {
                       PyObject** items = PySequence_Fast_ITEMS(sequences[i]);
                       const Py_ssize_t n = PySequence_Fast_GET_SIZE(sequences[i]);
                       for (Py_ssize_t j = 0; j < n; ++j) {
                           PyObject* str_obj = items[j];
                           if (PyUnicode_IS_ASCII(str_obj)) {
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(PyUnicode_1BYTE_DATA(str_obj));
                               lengths[pos] = static_cast<size_t>(PyUnicode_GET_LENGTH(str_obj));
                           } else {
                               Py_ssize_t size = 0;
                               const char* s = PyUnicode_AsUTF8AndSize(str_obj, &size);
                               if (!s) { had_error = true; break; }
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(s);
                               lengths[pos] = static_cast<size_t>(size);
                           }
                           ++pos;
                       }
                   }
                   if (had_error) {
                       for (auto* seq : sequences) Py_DECREF(seq);
                       throw py::error_already_set();
                   }
                   for (auto* seq : sequences) Py_DECREF(seq);
               }
               else if (is_bytes_data) {
                   for (size_t i = 0; i < B; ++i) {
                       PyObject** items = PySequence_Fast_ITEMS(sequences[i]);
                       const Py_ssize_t n = PySequence_Fast_GET_SIZE(sequences[i]);
                       for (Py_ssize_t j = 0; j < n; ++j) {
                           PyObject* it = items[j];
                           if (PyBytes_CheckExact(it)) {
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(it));
                               lengths[pos] = static_cast<size_t>(PyBytes_GET_SIZE(it));
                           } else {
                               for (auto* seq : sequences) Py_DECREF(seq);
                               throw py::value_error("All items must be bytes");
                           }
                           ++pos;
                       }
                   }
                   for (auto* seq : sequences) Py_DECREF(seq);
               }
               else {
                   for (size_t i = 0; i < B; ++i) {
                       PyObject** items = PySequence_Fast_ITEMS(sequences[i]);
                       const Py_ssize_t n = PySequence_Fast_GET_SIZE(sequences[i]);
                       for (Py_ssize_t j = 0; j < n; ++j) {
                           PyObject* it = items[j];
                           if (PyUnicode_Check(it)) {
                               Py_ssize_t size = 0;
                               const char* s = PyUnicode_AsUTF8AndSize(it, &size);
                               if (!s) {
                                   for (auto* seq : sequences) Py_DECREF(seq);
                                   throw py::error_already_set();
                               }
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(s);
                               lengths[pos] = static_cast<size_t>(size);
                           } else if (PyBytes_Check(it)) {
                               char* data = nullptr; Py_ssize_t size = 0;
                               if (PyBytes_AsStringAndSize(it, &data, &size) == -1) {
                                   for (auto* seq : sequences) Py_DECREF(seq);
                                   throw py::error_already_set();
                               }
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(data);
                               lengths[pos] = static_cast<size_t>(size);
                           } else if (PyByteArray_Check(it)) {
                               char* data = PyByteArray_AsString(it);
                               Py_ssize_t size = PyByteArray_Size(it);
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(data);
                               lengths[pos] = static_cast<size_t>(size);
                           } else if (PyObject_CheckBuffer(it)) {
                               Py_buffer view;
                               if (PyObject_GetBuffer(it, &view, PyBUF_SIMPLE) == -1) {
                                   for (auto* seq : sequences) Py_DECREF(seq);
                                   throw py::error_already_set();
                               }
                               ptrs[pos] = reinterpret_cast<const uint8_t*>(view.buf);
                               lengths[pos] = static_cast<size_t>(view.len);
                               retained_buffers.push_back(view);
                           } else {
                               for (auto* seq : sequences) Py_DECREF(seq);
                               throw py::value_error("All inner items must be str/bytes/bytearray or buffer");
                           }
                           ++pos;
                       }
                   }
                   for (auto* seq : sequences) Py_DECREF(seq);
               }
               std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_bytes(ptrs.get(), lengths.get(), indptr.data(), B, flat.get(), num_threads);
               }
               for (auto& v : retained_buffers) { PyBuffer_Release(&v); }
               return wrap_flat_as_2d(flat.release(), B, t);
           }

           // Integer iterable fast path: build CSR and return np.ndarray (B,t)
           {
               size_t total_n = 0;
               std::vector<uint64_t> indptr_vec; indptr_vec.reserve(B + 1);
               indptr_vec.push_back(0);
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
               return wrap_flat_as_2d(flat.release(), B, t);
           }
       }, py::arg("batches"), py::arg("prehashed") = false, py::arg("num_threads") = 0,
          "Compute sketches for a batch.\n"
          "  prehashed=False: batches are list of token sets (arrays/lists of ints or bytes/str).\n"
          "  prehashed=True: batches are list of pre-hashed uint64 arrays.\n"
          "  num_threads: 0 uses all threads (if OpenMP enabled). 1 forces single-thread.")

      // ── batch_csr(data, indptr, prehashed=False, num_threads=0) ───────
      .def("batch_csr", [](FastSimilaritySketch& self,
                            py::object data_obj,
                            py::array_t<uint64_t, py::array::c_style | py::array::forcecast> indptr,
                            bool prehashed,
                            int num_threads) {
           py::buffer_info bi = indptr.request();
           if (bi.ndim != 1) throw py::value_error("indptr must be a 1D array");
           if (bi.size < 2) throw py::value_error("indptr must have length >= 2");
           const size_t B = static_cast<size_t>(bi.size - 1);
           const size_t t = static_cast<size_t>(self.t);
           uint64_t* iptr = static_cast<uint64_t*>(bi.ptr);

           if (prehashed) {
               // Prehashed: data is uint64
               auto data = py::cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(data_obj);
               py::buffer_info bd = data.request();
               if (bd.ndim != 1) throw py::value_error("data must be a 1D array");
               const uint64_t* dptr = static_cast<const uint64_t*>(bd.ptr);
               std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_csr_prehashed(dptr, iptr, B, flat.get(), num_threads);
               }
               return wrap_flat_as_2d(flat.release(), B, t);
           } else {
               // Non-prehashed: data is uint32
               auto data = py::cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(data_obj);
               py::buffer_info bd = data.request();
               if (bd.ndim != 1) throw py::value_error("data must be a 1D array");
               uint32_t* dptr = static_cast<uint32_t*>(bd.ptr);
               std::unique_ptr<uint64_t[]> flat(new uint64_t[B * t]);
               {
                   py::gil_scoped_release release;
                   self.sketch_batch_flat_csr(dptr, iptr, B, flat.get(), num_threads);
               }
               return wrap_flat_as_2d(flat.release(), B, t);
           }
      }, py::arg("data"), py::arg("indptr"), py::arg("prehashed") = false, py::arg("num_threads") = 0,
         "CSR batch sketch.\n"
         "  prehashed=False: data is np.uint32, indptr is np.uint64 (length B+1).\n"
         "  prehashed=True:  data is np.uint64 (pre-hashed), indptr is np.uint64.\n"
         "Returns np.ndarray (B, t) of uint64.")
    ;

    // ===================== estimate_jaccard =====================

    m.def(
        "estimate_jaccard",
        [](py::object sketch_a, py::object sketch_b) {
            py::array_t<std::uint64_t> arr_a_handle;
            py::array_t<std::uint64_t> arr_b_handle;
            const std::uint64_t* ptr_a = nullptr;
            const std::uint64_t* ptr_b = nullptr;
            size_t size_a = 0;
            size_t size_b = 0;
            thread_local std::vector<std::uint64_t> scratch_a;
            thread_local std::vector<std::uint64_t> scratch_b;

            auto bind_view = [](py::object obj,
                                py::array_t<std::uint64_t>& arr_handle,
                                std::vector<std::uint64_t>& tmp,
                                const std::uint64_t*& ptr,
                                size_t& size) {
                if (py::isinstance<py::array_t<std::uint64_t>>(obj)) {
                    arr_handle = py::cast<py::array_t<std::uint64_t>>(obj);
                    py::buffer_info buf = arr_handle.request();
                    if (buf.ndim != 1) {
                        throw py::value_error("Sketch arrays must be 1-dimensional");
                    }
                    ptr = static_cast<const std::uint64_t*>(buf.ptr);
                    size = static_cast<size_t>(buf.size);
                    return;
                }

                PyObject* seq = PySequence_Fast(obj.ptr(), "Sketch must be a sequence of integers");
                if (!seq) {
                    throw py::error_already_set();
                }
                const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                tmp.resize(static_cast<size_t>(n));
                PyObject** items = PySequence_Fast_ITEMS(seq);
                for (Py_ssize_t i = 0; i < n; ++i) {
                    unsigned long long value = PyLong_AsUnsignedLongLong(items[i]);
                    if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
                        Py_DECREF(seq);
                        throw py::value_error("Sketch entries must fit in uint64");
                    }
                    tmp[static_cast<size_t>(i)] = static_cast<std::uint64_t>(value);
                }
                Py_DECREF(seq);
                ptr = tmp.data();
                size = tmp.size();
            };

            bind_view(sketch_a, arr_a_handle, scratch_a, ptr_a, size_a);
            bind_view(sketch_b, arr_b_handle, scratch_b, ptr_b, size_b);

            if (size_a != size_b) {
                throw py::value_error("Sketches must have identical length");
            }
            if (size_a == 0) {
                throw py::value_error("Sketch length must be greater than zero");
            }

            size_t matches = 0;
            for (size_t i = 0; i < size_a; ++i) {
                matches += (ptr_a[i] == ptr_b[i]) ? 1 : 0;
            }

            return static_cast<double>(matches) / static_cast<double>(size_a);
        },
        py::arg("sketch_a"),
        py::arg("sketch_b"),
        "Estimate Jaccard similarity between two 1-D uint64 sketches.");


    // ===================== LSH =====================

    py::class_<LSH>(m, "LSH")
      .def(py::init([](std::size_t num_perm, std::size_t num_bands,
                       std::uint64_t seed, int num_threads) {
               return new LSH(num_perm, num_bands,
                              LSH::BandHashKind::splitmix64, seed, num_threads);
           }),
           py::arg("num_perm"),
           py::arg("num_bands"),
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

      // ── insert (3 overloads) ──────────────────────────────────────────

      // Insert from 2D NumPy ndarray (B, t), dtype=uint64
      .def("insert", [](LSH& self,
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
       }, py::arg("data"),
          "Insert from 2D NumPy ndarray (uint64) with zero/low-copy")

      // Insert from list of 1D NumPy arrays
      .def("insert", [](LSH& self, py::list rows) {
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
          "Insert from list of NumPy arrays (uint64, length t)")

      // Insert from list of Python lists
      .def("insert", [](LSH& self, py::object py_rows) {
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
          "Insert from list of Python lists (copied once into a temporary buffer)")

      // ── query(input, format=None) ─────────────────────────────────────
      .def("query", [](const LSH& self, py::object input, py::object format) -> py::object {
           PyObject* obj_ptr = input.ptr();

           // ── 2D NumPy array → batch mode ──
           if (PyArray_Check(obj_ptr)) {
               PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj_ptr);
               int ndim = PyArray_NDIM(arr);

               if (ndim == 2) {
                   auto typed = py::cast<py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>>(input);
                   py::buffer_info bi = typed.request();
                   const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
                   const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
                   const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);

                   // Check format
                   bool use_csr = false;
                   if (!format.is_none()) {
                       std::string fmt = py::cast<std::string>(format);
                       if (fmt == "csr") use_csr = true;
                       else throw py::value_error("format must be None or 'csr'");
                   }

                   std::vector<std::size_t> flat;
                   std::vector<std::uint64_t> indptr;
                   {
                       py::gil_scoped_release release;
                       self.query_candidates_batch(base, B, t, flat, indptr);
                   }

                   if (use_csr) {
                       // Return (flat uint64 array, indptr uint64 array)
                       const ssize_t nflat = static_cast<ssize_t>(flat.size());
                       std::uint64_t* flat_raw = new std::uint64_t[static_cast<std::size_t>(nflat)];
                       for (ssize_t i = 0; i < nflat; ++i)
                           flat_raw[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(flat[static_cast<std::size_t>(i)]);
                       py::capsule owner_flat(flat_raw, [](void* f){ delete[] reinterpret_cast<std::uint64_t*>(f); });
                       py::array flat_arr(
                           py::dtype::of<std::uint64_t>(),
                           std::vector<ssize_t>{nflat},
                           std::vector<ssize_t>{static_cast<ssize_t>(sizeof(std::uint64_t))},
                           flat_raw, owner_flat
                       );
                       const ssize_t nip = static_cast<ssize_t>(indptr.size());
                       std::uint64_t* ip_raw = new std::uint64_t[static_cast<std::size_t>(nip)];
                       for (ssize_t i = 0; i < nip; ++i)
                           ip_raw[static_cast<std::size_t>(i)] = indptr[static_cast<std::size_t>(i)];
                       py::capsule owner_ip(ip_raw, [](void* f){ delete[] reinterpret_cast<std::uint64_t*>(f); });
                       py::array indptr_arr(
                           py::dtype::of<std::uint64_t>(),
                           std::vector<ssize_t>{nip},
                           std::vector<ssize_t>{static_cast<ssize_t>(sizeof(std::uint64_t))},
                           ip_raw, owner_ip
                       );
                       return py::make_tuple(flat_arr, indptr_arr);
                   } else {
                       // Return list[list[int]]
                       py::list outer(static_cast<py::ssize_t>(B));
                       for (std::size_t i = 0; i < B; ++i) {
                           const std::size_t start = static_cast<std::size_t>(indptr[i]);
                           const std::size_t end = static_cast<std::size_t>(indptr[i+1]);
                           const std::size_t len = end - start;
                           py::list inner(static_cast<py::ssize_t>(len));
                           for (std::size_t j = 0; j < len; ++j) {
                               const std::size_t id = flat[start + j];
                               PyObject* pyint = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(id));
                               PyList_SET_ITEM(inner.ptr(), static_cast<Py_ssize_t>(j), pyint);
                           }
                           PyList_SET_ITEM(outer.ptr(), static_cast<Py_ssize_t>(i), inner.release().ptr());
                       }
                       return py::cast<py::object>(outer);
                   }
               }
               else if (ndim == 1) {
                   // ── 1D NumPy array → single query ──
                   auto typed = py::cast<py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>>(input);
                   py::buffer_info bi = typed.request();
                   const std::size_t t = static_cast<std::size_t>(bi.size);
                   const std::uint64_t* ptr = static_cast<const std::uint64_t*>(bi.ptr);
                   std::vector<std::size_t> result;
                   {
                       py::gil_scoped_release release;
                       result = self.query_candidates(ptr, t);
                   }
                   // Convert to Python list of ints
                   py::list out(static_cast<py::ssize_t>(result.size()));
                   for (std::size_t i = 0; i < result.size(); ++i) {
                       PyObject* pyint = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(result[i]));
                       PyList_SET_ITEM(out.ptr(), static_cast<Py_ssize_t>(i), pyint);
                   }
                   return py::cast<py::object>(out);
               }
               else {
                   throw py::value_error("NumPy input must be 1D (single query) or 2D (batch query)");
               }
           }

           // ── Iterable → single query ──
           std::vector<std::uint64_t> buf;
           buf.reserve(self.num_perm());
           for (auto item : py::cast<py::iterable>(input)) {
               unsigned long long v = PyLong_AsUnsignedLongLong(item.ptr());
               if (v == (unsigned long long)-1 && PyErr_Occurred())
                   throw py::value_error("All items must be integers");
               buf.push_back(static_cast<std::uint64_t>(v));
           }
           std::vector<std::size_t> result;
           {
               py::gil_scoped_release release;
               result = self.query_candidates(buf.data(), buf.size());
           }
           py::list out(static_cast<py::ssize_t>(result.size()));
           for (std::size_t i = 0; i < result.size(); ++i) {
               PyObject* pyint = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(result[i]));
               PyList_SET_ITEM(out.ptr(), static_cast<Py_ssize_t>(i), pyint);
           }
           return py::cast<py::object>(out);
       }, py::arg("input"), py::arg("format") = py::none(),
          "Query candidates.\n"
          "  1D array or iterable → single query → list[int]\n"
          "  2D array + format=None → batch query → list[list[int]]\n"
          "  2D array + format='csr' → batch CSR → (flat, indptr)")

      // ── duplicates(arr, self_start=0) ─────────────────────────────────
      .def("duplicates", [](const LSH& self,
                            py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> arr,
                            std::uint64_t self_start) {
           py::buffer_info bi = arr.request();
           if (bi.ndim != 2) throw py::value_error("Input must be 2D (B,t) uint64 array");
           const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
           const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
           const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);

           std::vector<std::uint8_t> flags;
           {
               py::gil_scoped_release release;
               self.query_duplicate_flags_batch(
                   base, B, t,
                   static_cast<std::size_t>(self_start),
                   flags
               );
           }

           py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(B));
           auto out_view = out.mutable_unchecked<1>();
           for (std::size_t i = 0; i < B; ++i) {
               out_view(static_cast<py::ssize_t>(i)) = flags[i];
           }
           return out;
       }, py::arg("data"), py::arg("self_start") = 0,
          "Batch query returning duplicate flags as uint8 for self-query batches")

      // ── insert_and_query_duplicates(data) ─────────────────────────────
      .def("insert_and_query_duplicates", [](LSH& self,
                                             py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> arr) {
           py::buffer_info bi = arr.request();
           if (bi.ndim != 2) throw py::value_error("Input must be 2D (B,t) uint64 array");
           const std::size_t B = static_cast<std::size_t>(bi.shape[0]);
           const std::size_t t = static_cast<std::size_t>(bi.shape[1]);
           const std::uint64_t* base = static_cast<const std::uint64_t*>(bi.ptr);

           std::vector<std::uint8_t> flags;
           {
               py::gil_scoped_release release;
               self.insert_and_query_duplicate_flags_batch(base, B, t, flags);
           }

           py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(B));
           auto out_view = out.mutable_unchecked<1>();
           for (std::size_t i = 0; i < B; ++i) {
               out_view(static_cast<py::ssize_t>(i)) = flags[i];
           }
           return out;
       }, py::arg("data"),
          "Insert a 2D uint64 sketch matrix and return duplicate flags for the inserted rows")

      .def("insert_and_query_duplicates", [](LSH& self, py::list rows) {
           const std::size_t B = static_cast<std::size_t>(rows.size());
           const std::size_t t = self.num_perm();
           std::unique_ptr<const std::uint64_t*[]> ptrs(new const std::uint64_t*[B]);
           for (std::size_t i = 0; i < B; ++i) {
               auto arr = py::cast<py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>>(rows[i]);
               py::buffer_info bi = arr.request();
               if (bi.ndim != 1) throw py::value_error("Each array must be 1D");
               if (static_cast<std::size_t>(bi.size) != t) throw py::value_error("Row length must equal num_perm");
               ptrs[i] = static_cast<const std::uint64_t*>(bi.ptr);
           }

           std::vector<std::uint8_t> flags;
           {
               py::gil_scoped_release release;
               self.insert_and_query_duplicate_flags_batch(ptrs.get(), B, t, flags);
           }

           py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(B));
           auto out_view = out.mutable_unchecked<1>();
           for (std::size_t i = 0; i < B; ++i) {
               out_view(static_cast<py::ssize_t>(i)) = flags[i];
           }
           return out;
       }, py::arg("rows"),
          "Insert a list of uint64 sketch rows and return duplicate flags for the inserted rows")

      .def("insert_and_query_duplicates", [](LSH& self, py::object py_rows) {
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

           std::vector<std::uint8_t> flags;
           {
               py::gil_scoped_release release;
               self.insert_and_query_duplicate_flags_batch(base, B, t, flags);
           }

           py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(B));
           auto out_view = out.mutable_unchecked<1>();
           for (std::size_t i = 0; i < B; ++i) {
               out_view(static_cast<py::ssize_t>(i)) = flags[i];
           }
           return out;
       }, py::arg("rows"),
          "Insert Python integer rows and return duplicate flags for the inserted rows")

      // Read-only properties
      .def_property_readonly("num_perm",  &LSH::num_perm)
      .def_property_readonly("num_bands", &LSH::num_bands)
      .def_property_readonly("band_size", &LSH::band_size)
    ;
}
