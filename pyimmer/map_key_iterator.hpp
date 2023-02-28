#pragma once

#include "types.hpp"

#include <Python.h>

namespace pyimmer::map_key_iterator {

static PyObject *py_immer_map_key_iterator_new(PyTypeObject *type,
                                               PyObject *args,
                                               PyObject *kwargs) {
  PyObject *py_immer_map = PyTuple_GET_ITEM(args, 0);

  py_immer_map_key_iterator_t *py_immer_map_key_iterator =
      (py_immer_map_key_iterator_t *)type->tp_alloc(type, 0);

  if (not py_immer_map_key_iterator) {
    return NULL;
  }

  Py_INCREF(py_immer_map);
  py_immer_map_key_iterator->py_immer_map =
      (pyimmer::map::py_immer_map_t *)py_immer_map;
  auto &immer_map = py_immer_map_key_iterator->py_immer_map->immer_map;
  py_immer_map_key_iterator->iterator = immer_map.begin();
  py_immer_map_key_iterator->index = 0;
  py_immer_map_key_iterator->length = immer_map.size();

  return (PyObject *)py_immer_map_key_iterator;
}

static PyObject *py_immer_map_key_iterator_next(
    py_immer_map_key_iterator_t *py_immer_map_key_iterator) {

  if (py_immer_map_key_iterator->index < py_immer_map_key_iterator->length) {
    PyObject *key = py_immer_map_key_iterator->iterator->first.get();
    PyObject *result = Py_BuildValue("O", key);

    py_immer_map_key_iterator->iterator =
        std::next(py_immer_map_key_iterator->iterator);
    py_immer_map_key_iterator->index++;

    return result;
  }
  py_immer_map_key_iterator->index = py_immer_map_key_iterator->length;
  return NULL;
}

static void py_immer_map_key_iterator_dealloc(
    py_immer_map_key_iterator_t *py_immer_map_key_iterator) {

  Py_INCREF(py_immer_map_key_iterator->py_immer_map);
  Py_TYPE(py_immer_map_key_iterator)->tp_free(py_immer_map_key_iterator);
}

PyTypeObject py_immer_map_key_iterator_type = {
    // clang-format off
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    // clang-format on
    "py_immer_map_key_iterator",                   /* tp_name */
    sizeof(py_immer_map_key_iterator_t),           /* tp_basicsize */
    0,                                             /* tp_itemsize */
    (destructor)py_immer_map_key_iterator_dealloc, /* tp_dealloc */
    0,                                             /* tp_print */
    0,                                             /* tp_getattr */
    0,                                             /* tp_setattr */
    0,                                             /* tp_reserved */
    0,                                             /* tp_repr */
    0,                                             /* tp_as_number */
    0,                                             /* tp_as_sequence */
    0,                                             /* tp_as_mapping */
    0,                                             /* tp_hash */
    0,                                             /* tp_call */
    0,                                             /* tp_str */
    0,                                             /* tp_getattro */
    0,                                             /* tp_setattro */
    0,                                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                            /* tp_flags */
    0,                                             /* tp_doc */
    0,                                             /* tp_traverse */
    0,                                             /* tp_clear */
    0,                                             /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    PyObject_SelfIter,                             /* tp_iter */
    (iternextfunc)py_immer_map_key_iterator_next,  /* tp_iternext */
    0,                                             /* tp_methods */
    0,                                             /* tp_members */
    0,                                             /* tp_getset */
    0,                                             /* tp_base */
    0,                                             /* tp_dict */
    0,                                             /* tp_descr_get */
    0,                                             /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    0,                                             /* tp_init */
    PyType_GenericAlloc,                           /* tp_alloc */
    py_immer_map_key_iterator_new,                 /* tp_new */
};

} // namespace pyimmer::map_key_iterator