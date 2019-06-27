set(MODULE_NAME "DataFormatsFITV0")
set(MODULE_BUCKET_NAME data_format_fit_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Hit.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Hit.h
    )

Set(LINKDEF src/DataFormatsFITV0LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()


