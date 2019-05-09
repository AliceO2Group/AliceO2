SET(MODULE_NAME CPVBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Geometry.cxx
  src/Hit.cxx
  src/Digit.cxx
 )

set(HEADERS
  include/${MODULE_NAME}/Geometry.h
  include/${MODULE_NAME}/Hit.h
  include/${MODULE_NAME}/Digit.h
)

SET(LINKDEF src/CPVBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME cpv_base_bucket)

O2_GENERATE_LIBRARY()
