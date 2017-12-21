set(MODULE_NAME "DataFormatsParameters")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/GRPObject.cxx
)

Set(HEADERS
    include/${MODULE_NAME}/GRPObject.h
)

Set(LINKDEF src/ParametersDataLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME data_parameters_bucket)

O2_GENERATE_LIBRARY()

