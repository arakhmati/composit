#include "map.hpp"
#include "map_key_iterator.hpp"

#include <Python.h>

static PyMethodDef pyimmer_module_methods[] = {
    {"pmap", (PyCFunction)pyimmer::map::py_immer_map, METH_VARARGS}, {NULL}};

constexpr auto module_size = -1;
static struct PyModuleDef pyimmer_module = {
    PyModuleDef_HEAD_INIT, "pyimmer",
    "Wrapper around C++ Immer Data Structures", module_size,
    pyimmer_module_methods};

PyMODINIT_FUNC PyInit_pyimmer(void) {

  PyObject *module = PyModule_Create(&pyimmer_module);
  if (module == NULL) {
    return NULL;
  }

  if (PyType_Ready(&pyimmer::map::py_immer_map_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&pyimmer::map::py_immer_map_type);
  PyModule_AddObject(module, "PMap",
                     (PyObject *)&pyimmer::map::py_immer_map_type);

  if (PyType_Ready(&pyimmer::map_key_iterator::py_immer_map_key_iterator_type) < 0) {
    return NULL;
  }
  Py_INCREF((PyObject *)&pyimmer::map_key_iterator::py_immer_map_key_iterator_type);
  PyModule_AddObject(
      module, pyimmer::map_key_iterator::py_immer_map_key_iterator_type.tp_name,
      (PyObject *)&pyimmer::map_key_iterator::py_immer_map_key_iterator_type);

  return module;
}
