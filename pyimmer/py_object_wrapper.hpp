#pragma once

#include <Python.h>

#include <functional>

namespace pyimmer::py_object_wrapper {
class py_object_wrapper_t;
}

template <> struct std::hash<pyimmer::py_object_wrapper::py_object_wrapper_t>;

namespace pyimmer::py_object_wrapper {
class py_object_wrapper_t {
public:
  py_object_wrapper_t(PyObject *py_object) : py_object_(py_object) {
    Py_INCREF(this->py_object_);
  }
  py_object_wrapper_t() { Py_DECREF(this->py_object_); }

  PyObject *get() const {
    Py_INCREF(this->py_object_);
    return this->py_object_;
  }

  friend struct std::hash<py_object_wrapper_t>;

private:
  PyObject *py_object_;
};

bool operator<(const py_object_wrapper_t &a,
               const py_object_wrapper_t &b) noexcept {
  return PyObject_RichCompareBool(a.get(), b.get(), Py_LT);
}

bool operator==(const py_object_wrapper_t &a,
                const py_object_wrapper_t &b) noexcept {
  return PyObject_RichCompareBool(a.get(), b.get(), Py_EQ);
}

} // namespace pyimmer::py_object_wrapper

template <> struct std::hash<pyimmer::py_object_wrapper::py_object_wrapper_t> {
  std::size_t operator()(const pyimmer::py_object_wrapper::py_object_wrapper_t
                             &object) const noexcept {
    return PyObject_Hash(object.get());
  }
};