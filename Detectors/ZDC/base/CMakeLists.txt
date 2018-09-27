SET(MODULE_NAME ZDCBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Geometry.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Geometry.h
)

SET(LINKDEF src/ZDCBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME zdc_base_bucket)

O2_GENERATE_LIBRARY()
