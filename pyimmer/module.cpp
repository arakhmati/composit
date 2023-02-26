#include "map.hpp"

#include <Python.h>

static PyMethodDef pyimmer_module_methods[] = {
    {"pmap", (PyCFunction)pyimmer::map::pmap, METH_VARARGS}, {NULL}};

constexpr auto module_size = -1;
static struct PyModuleDef pyimmer_module = {
    PyModuleDef_HEAD_INIT, "pyimmer",
    "Wrapper around C++ Immer Data Structures", module_size,
    pyimmer_module_methods};

PyMODINIT_FUNC PyInit_pyimmer(void) {

  PyObject *module;
  if (PyType_Ready(&pyimmer::map::py_immer_map_type) < 0)
    return NULL;

  module = PyModule_Create(&pyimmer_module);
  if (module == NULL)
    return NULL;

  Py_INCREF(&pyimmer::map::py_immer_map_type);
  if (PyModule_AddObject(module, "PMap",
                         (PyObject *)&pyimmer::map::py_immer_map_type) < 0) {
    Py_DECREF(&pyimmer::map::py_immer_map_type);
    Py_DECREF(module);
    return NULL;
  }

  return module;
}
