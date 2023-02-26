#include "memory_policy.hpp"
#include "py_object_wrapper.hpp"

#include <immer/map.hpp>

#include <Python.h>
#include <structmember.h>

namespace pyimmer::map {

using pyimmer::py_object_wrapper::py_object_wrapper_t;

using immer_map_t = immer::map<py_object_wrapper_t, py_object_wrapper_t,
                               std::hash<py_object_wrapper_t>,
                               std::equal_to<py_object_wrapper_t>,
                               pyimmer::memory_policy::memory_policy_t>;

typedef struct py_immer_map_t {
  PyObject_HEAD;
  immer_map_t immer_map;
} py_immer_map_t;

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

static PyObject *py_immer_map_get(PyObject *self, PyObject *args) {
  auto key = py_object_wrapper_t{PyTuple_GET_ITEM(args, 0)};
  auto pmap = (py_immer_map_t *)self;
  return pmap->immer_map.at(key).get();
}

static Py_ssize_t py_immer_map_len(PyObject *self) {
  auto pmap = (py_immer_map_t *)self;
  return pmap->immer_map.size();
}

static PyMappingMethods py_immer_map_mapping_methods = {
    .mp_length = py_immer_map_len,
    .mp_subscript = py_immer_map_get,
    .mp_ass_subscript = NULL,
};

static PyMethodDef py_immer_map_methods[] = {
    {"get", (PyCFunction)py_immer_map_get, METH_VARARGS,
     "Return the value given key"},
    {NULL} /* Sentinel */
};

static PyMemberDef py_immer_map_members[] = {
    {NULL} /* Sentinel */
};

static PyTypeObject py_immer_map_type = {
    PyObject_HEAD_INIT(NULL) "pyimmer.PMap",
    sizeof(py_immer_map_t),
    0,
    {}, /* tp_dealloc */
    {}, /* tp_vectorcall_offset */
    {}, /* tp_getattr */
    {}, /* tp_setattr */
    {}, /* tp_as_async */
    {}, /* tp_repr */
    {}, /* tp_as_number */
    {}, /* tp_as_sequence */
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
    {}, /* tp_iter */
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

static py_immer_map_t *pmap(PyObject *self, PyObject *args) {

  auto pmap = (py_immer_map_t *)(PyObject_CallObject(
      (PyObject *)&py_immer_map_type, NULL));
  auto dictionary = PyTuple_GET_ITEM(args, 0);

  Py_ssize_t position = 0;
  PyObject *key, *value;
  while (PyDict_Next(dictionary, &position, &key, &value)) {
    pmap->immer_map = pmap->immer_map.insert(
        {py_object_wrapper_t{key}, py_object_wrapper_t{value}});
  }

  return pmap;
}

} // namespace pyimmer::map
