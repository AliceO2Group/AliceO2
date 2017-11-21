set(MODULE_NAME "Steer")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/O2RunAna.cxx
)

set(HEADERS
  include/${MODULE_NAME}/O2RunAna.h
)

set(LINKDEF src/SteerLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_simulation_bucket)

O2_GENERATE_LIBRARY()
