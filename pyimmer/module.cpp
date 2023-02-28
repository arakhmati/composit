#include "map.hpp"
#include "map_item_iterator.hpp"
#include "map_key_iterator.hpp"
#include "map_value_iterator.hpp"

#include <Python.h>

static PyMethodDef pyimmer_module_methods[] = {
    {"pmap", (PyCFunction)pyimmer::map::py_immer_map, METH_VARARGS}, {NULL}};

constexpr auto module_size = -1;
static struct PyModuleDef pyimmer_module = {
    PyModuleDef_HEAD_INIT, "pyimmer",
    "Wrapper around C++ Immer Data Structures", module_size,
    pyimmer_module_methods};

PyMODINIT_FUNC PyInit_pyimmer(void) {

  using pyimmer::map::py_immer_map_type;
  using pyimmer::map_key_iterator::py_immer_map_key_iterator_type;
  using pyimmer::map_value_iterator::py_immer_map_value_iterator_type;
  using pyimmer::map_item_iterator::py_immer_map_item_iterator_type;

  PyObject *module = PyModule_Create(&pyimmer_module);
  if (module == NULL) {
    return NULL;
  }

  if (PyType_Ready(&py_immer_map_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&py_immer_map_type);
  PyModule_AddObject(module, "PMap", (PyObject *)&py_immer_map_type);

  if (PyType_Ready(&py_immer_map_key_iterator_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&py_immer_map_key_iterator_type);
  PyModule_AddObject(module, py_immer_map_key_iterator_type.tp_name,
                     (PyObject *)&py_immer_map_key_iterator_type);

  if (PyType_Ready(&py_immer_map_value_iterator_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&py_immer_map_value_iterator_type);
  PyModule_AddObject(module, py_immer_map_value_iterator_type.tp_name,
                     (PyObject *)&py_immer_map_value_iterator_type);

  if (PyType_Ready(&py_immer_map_item_iterator_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&py_immer_map_item_iterator_type);
  PyModule_AddObject(module, py_immer_map_item_iterator_type.tp_name,
                     (PyObject *)&py_immer_map_item_iterator_type);

  return module;
}
