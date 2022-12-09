#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "head.h"
#include <iostream>
#include "numpy/arrayobject.h"

static PyObject *InsertionError;

inline float PyNumber2float(PyObject *obj)
{
    return (float)(PyFloat_Check(obj) ? PyFloat_AsDouble(obj) : PyLong_AsDouble(obj));
}

static PyObject *
insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities;
    PyObject *pyorder;
    bool converted = false;
    if (!PyArg_ParseTuple(args, "OO", &pycities, &pyorder))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder))
        return NULL;
    if (PyArray_NDIM(pycities) != 2 || PyArray_NDIM(pyorder) != 1)
        return NULL;
    if (PyArray_TYPE(pycities) != NPY_FLOAT32 || PyArray_TYPE(pyorder) != NPY_UINT32)
        return NULL;

    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    // for (unsigned i = 0; i < cities_count; i++)
    //     printf("(%f, %f)\n", cities[i * 2], cities[i * 2 + 1]);

    /* ---------------------------- random insertion ---------------------------- */
    TSPinstance tspi = TSPinstance(citycount, cities);
    Insertion ins = Insertion(&tspi);
    unsigned *output = ins.randomInsertion(order);
    double distance = ins.getdistance();

    /* ----------------------- convert output to PyObject ----------------------- */
    npy_intp dims = citycount;
    PyObject *returnarr = PyArray_SimpleNewFromData(1, &dims, NPY_UINT32, output);
    PyObject *returntuple = PyTuple_Pack(2, returnarr, PyFloat_FromDouble(distance));

    return returntuple;
}

static PyMethodDef InsertionMethods[] = {
    {"random", insertion_random, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef insertionmodule = {
    PyModuleDef_HEAD_INIT,
    "insertion",
    NULL,
    -1,
    InsertionMethods};

PyMODINIT_FUNC
PyInit_insertion(void)
{
    PyObject *m;
    m = PyModule_Create(&insertionmodule);
    if (m == NULL)
        return NULL;
    import_array();

    return m;
}
