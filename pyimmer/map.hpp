#pragma once

#include "map_item_iterator.hpp"
#include "map_key_iterator.hpp"
#include "map_value_iterator.hpp"
#include "py_object_wrapper.hpp"
#include "types.hpp"

#include <immer/map.hpp>

#include <Python.h>
#include <structmember.h>

namespace pyimmer::map {

using pyimmer::py_object_wrapper::py_object_wrapper_t;

static py_immer_map_t *create_map_instance();

static PyObject *py_immer_map_New(PyTypeObject *type, PyObject *args,
                                  PyObject *kwds) {
  constexpr auto num_items = 0;
  auto *self = (py_immer_map_t *)type->tp_alloc(type, num_items);
  self->immer_map = {};
  return (PyObject *)self;
}

static int py_immer_map_Init(py_immer_map_t *self, PyObject *args,
                             PyObject *kwds) {
  return 0;
}

static PyObject *py_immer_map_get_item(PyObject *self, PyObject *key) {
  auto py_immer_map = (py_immer_map_t *)self;
  if (py_immer_map->immer_map.find(py_object_wrapper_t{key}) == nullptr) {
    PyErr_SetString(PyExc_KeyError, "Key is not in the persistent map!");
    return NULL;
  }
  return py_immer_map->immer_map.at(key).get();
}

static PyObject *py_immer_map_get(PyObject *self, PyObject *args) {

  auto nargs = PyTuple_Size(args);

  auto py_immer_map = (py_immer_map_t *)self;
  auto key = py_object_wrapper_t{PyTuple_GET_ITEM(args, 0)};

  PyObject *default_value = Py_None;
  if (nargs > 1) {
    default_value = PyTuple_GET_ITEM(args, 1);
  }

  if (py_immer_map->immer_map.find(key) == nullptr) {
    Py_INCREF(default_value);
    return default_value;
  }
  return py_immer_map->immer_map.at(key).get();
}

static Py_ssize_t py_immer_map_len(PyObject *self) {
  auto py_immer_map = (py_immer_map_t *)self;
  return py_immer_map->immer_map.size();
}

static int py_immer_map_contains(PyObject *self, PyObject *value) {

  auto py_immer_map = (py_immer_map_t *)self;
  auto key = py_object_wrapper_t{value};

  if (py_immer_map->immer_map.find(key) == nullptr) {
    return 0;
  }
  return 1;
}

static PySequenceMethods py_immer_sequence_methods = {
    .sq_length = py_immer_map_len,
    .sq_concat = NULL,
    .sq_repeat = NULL,
    .sq_item = NULL,
    .sq_ass_item = NULL,
    .sq_contains = py_immer_map_contains,
    .sq_inplace_concat = NULL,
    .sq_inplace_repeat = NULL,
};

static PyMappingMethods py_immer_map_mapping_methods = {
    .mp_length = py_immer_map_len,
    .mp_subscript = py_immer_map_get_item,
    .mp_ass_subscript = NULL,
};

static py_immer_map_t *py_immer_map_set(py_immer_map_t *self, PyObject *args) {

  auto old_pmap = (py_immer_map_t *)self;
  auto py_immer_map = create_map_instance();
  py_immer_map->immer_map = old_pmap->immer_map;

  auto key = PyTuple_GET_ITEM(args, 0);
  auto value = PyTuple_GET_ITEM(args, 1);
  py_immer_map->immer_map = py_immer_map->immer_map.insert(
      {py_object_wrapper_t{key}, py_object_wrapper_t{value}});

  return py_immer_map;
}

static py_immer_map_t *py_immer_map_update(py_immer_map_t *self,
                                           PyObject *args) {

  auto old_pmap = (py_immer_map_t *)self;
  auto py_immer_map = create_map_instance();
  py_immer_map->immer_map = old_pmap->immer_map;

  auto dictionary = PyTuple_GET_ITEM(args, 0);
  if (PyDict_CheckExact(dictionary)) {
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(dictionary, &position, &key, &value)) {
      py_immer_map->immer_map = py_immer_map->immer_map.insert(
          {py_object_wrapper_t{key}, py_object_wrapper_t{value}});
    }
  } else {
    auto immer_pmap_with_updates = ((py_immer_map_t *)dictionary)->immer_map;
    for (auto &&[key, value] : immer_pmap_with_updates) {
      py_immer_map->immer_map = py_immer_map->immer_map.insert({key, value});
    }
  }

  return py_immer_map;
}

static PyObject *py_immer_map_key_iter(PyObject *object) {

  using pyimmer::map_key_iterator::py_immer_map_key_iterator_t;
  using pyimmer::map_key_iterator::py_immer_map_key_iterator_type;
  PyObject *args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, object);

  return PyObject_CallObject((PyObject *)&py_immer_map_key_iterator_type, args);
}

static PyObject *py_immer_map_value_iter(PyObject *object) {

  using pyimmer::map_value_iterator::py_immer_map_value_iterator_t;
  using pyimmer::map_value_iterator::py_immer_map_value_iterator_type;
  PyObject *args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, object);

  return PyObject_CallObject((PyObject *)&py_immer_map_value_iterator_type,
                             args);
}

static PyObject *py_immer_map_item_iter(PyObject *object) {

  using pyimmer::map_item_iterator::py_immer_map_item_iterator_t;
  using pyimmer::map_item_iterator::py_immer_map_item_iterator_type;
  PyObject *args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, object);

  return PyObject_CallObject((PyObject *)&py_immer_map_item_iterator_type,
                             args);
}

static PyMethodDef py_immer_map_methods[] = {
    {"get", (PyCFunction)py_immer_map_get, METH_VARARGS,
     "Return value given key"},
    {"set", (PyCFunction)py_immer_map_set, METH_VARARGS,
     "Update map with key-value pair"},
    {"update", (PyCFunction)py_immer_map_update, METH_VARARGS,
     "Update map with key-value pairs from another map or dict"},
    {"keys", (PyCFunction)py_immer_map_key_iter, METH_VARARGS,
     "Iterate over keys"},
    {"values", (PyCFunction)py_immer_map_value_iter, METH_VARARGS,
     "Iterate over values"},
    {"items", (PyCFunction)py_immer_map_item_iter, METH_VARARGS,
     "Iterate over items"},
    {NULL} /* Sentinel */
};

static PyMemberDef py_immer_map_members[] = {
    {NULL} /* Sentinel */
};

static PyTypeObject py_immer_map_type = {
    // clang-format off
    PyObject_HEAD_INIT(NULL)
    // clang-format on
    "pyimmer.PMap",
    sizeof(py_immer_map_t),
    0,
    {}, /* tp_dealloc */
    {}, /* tp_vectorcall_offset */
    {}, /* tp_getattr */
    {}, /* tp_setattr */
    {}, /* tp_as_async */
    {}, /* tp_repr */
    {}, /* tp_as_number */
    &py_immer_sequence_methods,
    &py_immer_map_mapping_methods,
    {}, /* tp_hash */
    {}, /* tp_call */
    {}, /* tp_str */
    {}, /* tp_getattro */
    {}, /* tp_setattro */
    {}, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    PyDoc_STR("Python Wrapper of Immer Map"),
    {}, /* tp_traverse */
    {}, /* tp_clear */
    {}, /* tp_richcompare */
    {}, /* tp_weaklistoffset */
    py_immer_map_key_iter,
    {}, /* tp_iternext */
    py_immer_map_methods,
    py_immer_map_members,
    {}, /* tp_getset */
    {}, /* tp_base */
    {}, /* tp_dict */
    {}, /* tp_descr_get */
    {}, /* tp_descr_set */
    {}, /* tp_dictoffset */
    (initproc)py_immer_map_Init,
    {}, /* tp_alloc */
    py_immer_map_New,
};

static py_immer_map_t *create_map_instance() {
  return (py_immer_map_t *)(PyObject_CallObject((PyObject *)&py_immer_map_type,
                                                NULL));
}

static py_immer_map_t *py_immer_map(PyObject *self, PyObject *args) {

  auto py_immer_map = create_map_instance();

  auto nargs = PyTuple_Size(args);
  if (nargs > 0) {
    py_immer_map = py_immer_map_update(py_immer_map, args);
  }

  return py_immer_map;
}

} // namespace pyimmer::map
