#define PY_SSIZE_T_CLEAN
#include "Framework/StructToTuple.h"
#include <Python.h>
#include "structmember.h"

#include <functional>
#include <vector>
#include "Framework/Pack.h"

extern "C" {
void fillModuleDef(char const* name, PyCFunction func, int flags, char const* doc);
void fillSchema(char const* name, PyObject* schema);

typedef struct {
  PyObject_HEAD
    PyObject* columns;        /* all the columns*/
  PyObject* persistedColumns; /* the columns on disk*/
} PyO2TableSchema;

static void
  PyO2TableSchema_dealloc(PyO2TableSchema* self)
{
  Py_XDECREF(self->columns);
  Py_XDECREF(self->persistedColumns);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
  PyO2TableSchema_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  PyO2TableSchema* self;
  self = (PyO2TableSchema*)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->columns = PyDict_New();
    if (self->columns == NULL) {
      Py_DECREF(self);
      return NULL;
    }
    self->persistedColumns = PyDict_New();
    if (self->persistedColumns == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject*)self;
}

static PyMemberDef PyO2TableSchemaType_members[] = {
  {"columns", T_OBJECT_EX, offsetof(PyO2TableSchema, columns), 0,
   "all available columns"},
  {"persistentColumns", T_OBJECT_EX, offsetof(PyO2TableSchema, persistedColumns), 0,
   "all persisted columns"},
  {NULL} /* Sentinel */
};

static PyTypeObject PyO2TableSchemaType = {
  PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "O2TableSchema",
  .tp_basicsize = sizeof(PyO2TableSchema),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor)PyO2TableSchema_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc = PyDoc_STR("Schema for an O2 Table"),
  .tp_members = PyO2TableSchemaType_members,
  .tp_new = PyO2TableSchema_new,
};
}

namespace o2::soa
{
template <typename... Tables>
struct Join {
  struct iterator {
  };
};

template <typename... Tables>
struct Index {
  static char const* label() { return "Index"; }
  struct iterator {
  };
};
} // namespace o2::soa
#define DPL_CUSTOM_DATAMODEL()
#define DECLARE_SOA_STORE()
#define DECLARE_SOA_TABLE(_Table_, origin, desc, ...)                                                 \
  struct _Table_ {                                                                                    \
    using columns_t = std::tuple<__VA_ARGS__>;                                                        \
    _Table_()                                                                                         \
    {                                                                                                 \
      static PyObject* d = PyDict_New();                                                              \
      fillSchema(#_Table_, d);                                                                        \
      std::apply([](auto... x) { std::make_tuple(registerColumn<decltype(x)>(d)...); }, columns_t{}); \
    }                                                                                                 \
                                                                                                      \
    template <typename C>                                                                             \
    static int registerColumn(PyObject* d)                                                            \
    {                                                                                                 \
      PyDict_SetItem(d, Py_BuildValue("s", C::label()), Py_BuildValue("i", 123));                     \
      return 1;                                                                                       \
    }                                                                                                 \
    struct iterator {                                                                                 \
    };                                                                                                \
  };                                                                                                  \
  static _Table_ g_table_##_Table_;

#define DECLARE_SOA_TABLE_VERSIONED(table, origin, desc, vers, ...) \
  struct table {                                                    \
    struct iterator {                                               \
    };                                                              \
  };
#define DECLARE_SOA_TABLE_FULL_VERSIONED(table, label, origin, desc, vers, ...) \
  struct table {                                                                \
    struct iterator {                                                           \
    };                                                                          \
  };

#define DECLARE_SOA_TABLE_FULL(table, label, origin, desc, ...) \
  struct table {                                                \
    struct iterator {                                           \
    };                                                          \
  };

#define DECLARE_SOA_COLUMN_FULL(_Name_, _Method_, type, _Label_) \
  struct _Name_ {                                                \
    struct iterator {                                            \
    };                                                           \
    static char const* persistentLabel() { return #_Label_; }    \
    static char const* label() { return #_Method_; }             \
  };

#define DECLARE_SOA_COLUMN(_Name_, _Getter_, _Type_) \
  DECLARE_SOA_COLUMN_FULL(_Name_, _Getter_, _Type_, "f" #_Name_)

#define DECLARE_SOA_INDEX(_Table_, _Column_) \
  struct _Table_ {                           \
  };
#define DECLARE_SOA_SLICE_INDEX_COLUMN(_Column_, _Getter_, ...) \
  struct _Column_##IdSlice {                                    \
    static char const* label() { return #_Getter_ "IdSlice"; }  \
  };
#define DECLARE_SOA_INDEX_TABLE_EXCLUSIVE(...)
#define DECLARE_SOA_INDEX_TABLE(_Table_, ...) \
  struct _Table_ {                            \
    struct iterator {                         \
    };                                        \
  };
#define DECLARE_SOA_EXPRESSION_COLUMN(column, method, type, expr, ...)
#define DECLARE_SOA_DYNAMIC_COLUMN(_Column_, _Method_, ...) \
  template <typename... BINDINGS>                           \
  struct _Column_ {                                         \
    static char const* label() { return #_Method_; }        \
  };

#define DECLARE_SOA_SELF_INDEX_COLUMN_FULL(_Column_, _Getter_, ...) \
  struct _Column_##Id {                                             \
    static char const* label() { return #_Getter_; }                \
  };

#define DECLARE_SOA_SELF_ARRAY_INDEX_COLUMN(...)
#define DECLARE_SOA_SELF_SLICE_INDEX_COLUMN(...)
#define DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, _Type_, _Table_, _Suffix_) \
  struct _Name_##Id {                                                              \
    static char const* label() { return #_Getter_ "Id"; }                          \
  };
#define DECLARE_SOA_INDEX_COLUMN(_Name_, _Getter_) DECLARE_SOA_INDEX_COLUMN_FULL(_Name_, _Getter_, int32_t, _Name_##s, "")
#define DECLARE_EQUIVALENT_FOR_INDEX(...)
#define DECLARE_SOA_EXTENDED_TABLE(_Table_, initial, label, ...) \
  struct _Table_##Extension {                                    \
    struct iterator {                                            \
    };                                                           \
  };                                                             \
  struct _Table_ {                                               \
    struct iterator {                                            \
    };                                                           \
  };

#include "Framework/AnalysisDataModel.h"

extern "C" {
static PyObject* O2Error;

static PyObject*
  o2_system(PyObject* self, PyObject* args)
{
  return PyDict_New();
}

static PyMethodDef O2Methods[512] = {
  {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef o2module = {
  PyModuleDef_HEAD_INIT,
  "o2",    /* name of module */
  nullptr, /* module documentation, may be NULL */
  -1,      /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
  O2Methods};

static struct {
  char const* name;
  PyObject* schema;
} O2Schema[512];

void fillModuleDef(char const* name, PyCFunction func, int flags, char const* doc)
{
  static int i = 0;
  O2Methods[i].ml_name = name ? strdup(name) : 0;
  O2Methods[i].ml_meth = (PyCFunction)o2_system;
  O2Methods[i].ml_flags = METH_VARARGS;
  O2Methods[i].ml_doc = doc ? strdup(doc) : 0;
  if (!name) {
    return;
  }
  i++;
}

void fillSchema(char const* name, PyObject* schema)
{
  static int i = 0;
  O2Schema[i].name = name ? strdup(name) : nullptr;
  O2Schema[i].schema = schema;
  if (!name) {
    return;
  }
  i++;
}

PyMODINIT_FUNC
  PyInit_o2(void)
{
  fillModuleDef(0, 0, 0, 0);
  fillSchema(0, 0);
  if (PyType_Ready(&PyO2TableSchemaType) < 0)
    return NULL;

  static PyObject* s_Module = nullptr;
  s_Module = PyModule_Create(&o2module);
  if (s_Module == NULL)
    return NULL;

  Py_INCREF(&PyO2TableSchemaType);
  if (PyModule_AddObject(s_Module, "TableSchema", (PyObject*)&PyO2TableSchemaType) < 0) {
    Py_DECREF(&PyO2TableSchemaType);
    Py_DECREF(s_Module);
    return NULL;
  }
  // Fill the schema in the module
  for (size_t i = 0; i < 512; i++) {
    if (O2Schema[i].name == nullptr) {
      break;
    }
    // PyObject* arglist = Py_BuildValue("(i)", arg);
    // PyObject* tableSchema = PyObject_CallObject(, arglist);

    // Py_CLEAR(O2Schema[i].schema);
    // Py_XINCREF(O2Schema[i].schema);
    // PyDict_SetAttrString(tableSchema, "columns", O2Schema[i].schema);
    PyObject* tableSchema = PyObject_CallObject((PyObject*)&PyO2TableSchemaType, Py_BuildValue("()"));
    Py_XINCREF(tableSchema);
    if (PyModule_AddObject(s_Module, O2Schema[i].name, tableSchema) < 0) {
      Py_XDECREF(tableSchema);
      Py_CLEAR(tableSchema);
      Py_DECREF(s_Module);
      return NULL;
    }
    Py_XINCREF(O2Schema[i].schema);
    PyObject_SetAttrString(tableSchema, "columns", O2Schema[i].schema);
  }

  return s_Module;
}
}
