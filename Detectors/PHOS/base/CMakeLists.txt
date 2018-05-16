SET(MODULE_NAME PHOSBase)

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

SET(LINKDEF src/PHOSBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME phos_base_bucket)

O2_GENERATE_LIBRARY()
