#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "head.h"
#include "numpy/arrayobject.h"

static PyObject *
insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities;
    PyObject *pyorder;
    PyObject *pyout;

    int isEuclidean = 1;
    if (!PyArg_ParseTuple(args, "OOpO", &pycities, &pyorder, &isEuclidean, &pyout))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrout = (PyArrayObject *)pyout;

    if (isEuclidean && PyArray_NDIM(pyarrcities) != 2 || PyArray_NDIM(pyarrorder) != 1 || PyArray_NDIM(pyarrout) != 1)
        return NULL;
    if (PyArray_TYPE(pyarrcities) != NPY_FLOAT32 || PyArray_TYPE(pyarrorder) != NPY_UINT32 || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return NULL;

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *out = (unsigned *)PyArray_DATA(pyarrout);
    // for (unsigned i = 0; i < cities_count; i++)
    //     printf("(%f, %f)\n", cities[i * 2], cities[i * 2 + 1]);
    TSPinstance *tspi;
    if(isEuclidean){
        tspi = new TSPinstanceEuclidean(citycount, cities);
    }else{
        if(citycount!=(unsigned)shape[1])
            return NULL;
        tspi = new TSPinstanceNonEuclidean(citycount, cities);
    }

    /* ---------------------------- random insertion ---------------------------- */
    Insertion ins = Insertion(tspi);
    ins.randomInsertion(order);
    float distance = ins.getResult(out);

    /* ----------------------- convert output to PyObject ----------------------- */
    PyObject *pyresult = PyFloat_FromDouble(distance);

    delete tspi;
    return pyresult;
}

static PyObject *
cvrp_insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pydemands;
    float depotx, depoty, exploration;
    unsigned capacity;
    // positions depotx depoty demands capacity order
    if (!PyArg_ParseTuple(args, "OffOIOf", &pycities, &depotx, &depoty, &pydemands, &capacity, &pyorder, &exploration))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder) || !PyArray_Check(pydemands))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrdemands = (PyArrayObject *)pydemands;

    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_NDIM(pyarrorder) != 1 || PyArray_NDIM(pyarrdemands) != 1)
        return NULL;
    if (PyArray_TYPE(pyarrcities) != NPY_FLOAT32 || PyArray_TYPE(pyarrorder) != NPY_UINT32 || PyArray_TYPE(pyarrdemands) != NPY_UINT32 )
        return NULL;

    // std::printf("running\n");

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *demands = (unsigned *)PyArray_DATA(pyarrdemands);
    float depotpos[2] = {depotx, depoty};

    /* ---------------------------- random insertion ---------------------------- */
    CVRPInstance cvrpi = CVRPInstance(citycount, cities, demands, depotpos, capacity);
    CVRPInsertion ins = CVRPInsertion(&cvrpi);

    // std::printf("insertion\n");
    CVRPReturn *result = ins.randomInsertion(order, exploration);
    // std::printf("insertion finished\n");
    /* ----------------------- convert output to PyObject ----------------------- */
    npy_intp dims = citycount, dims2 = result->routes;
    PyObject *returntuple = PyTuple_Pack(2, 
        PyArray_SimpleNewFromData(1, &dims, NPY_UINT32, result->order),
        PyArray_SimpleNewFromData(1, &dims2, NPY_UINT32, result->routesep)
    );

    return returntuple;
}

static PyMethodDef InsertionMethods[] = {
    {"random", insertion_random, METH_VARARGS, "Execute random insertion."},
    {"cvrp_random", cvrp_insertion_random, METH_VARARGS, "Execute random insertion for CVRP."},
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
