#pragma once

#include "memory_policy.hpp"
#include "py_object_wrapper.hpp"

#include <immer/map.hpp>

#include <Python.h>

using pyimmer::py_object_wrapper::py_object_wrapper_t;

using immer_map_t = immer::map<py_object_wrapper_t, py_object_wrapper_t,
                               std::hash<py_object_wrapper_t>,
                               std::equal_to<py_object_wrapper_t>,
                               pyimmer::memory_policy::memory_policy_t>;

namespace pyimmer::map {

struct py_immer_map_t {
  PyObject_HEAD;
  immer_map_t immer_map;
};
} // namespace pyimmer::map

namespace pyimmer::map_key_iterator {
struct py_immer_map_key_iterator_t {
  PyObject_HEAD;
  pyimmer::map::py_immer_map_t *py_immer_map;
  immer_map_t::iterator iterator;
  Py_ssize_t index;
  Py_ssize_t length;
};
} // namespace pyimmer::map_key_iterator

namespace pyimmer::map_value_iterator {
struct py_immer_map_value_iterator_t {
  PyObject_HEAD;
  pyimmer::map::py_immer_map_t *py_immer_map;
  immer_map_t::iterator iterator;
  Py_ssize_t index;
  Py_ssize_t length;
};
} // namespace pyimmer::map_value_iterator

namespace pyimmer::map_item_iterator {
struct py_immer_map_item_iterator_t {
  PyObject_HEAD;
  pyimmer::map::py_immer_map_t *py_immer_map;
  immer_map_t::iterator iterator;
  Py_ssize_t index;
  Py_ssize_t length;
};
} // namespace pyimmer::map_item_iterator