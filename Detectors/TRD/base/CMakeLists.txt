SET(MODULE_NAME TRDBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/TRDPadPlane.cxx
  src/TRDGeometry.cxx
  src/TRDCommonParam.cxx
  src/TRDSimParam.cxx
)

set(HEADERS
  include/${MODULE_NAME}/TRDPadPlane.h
  include/${MODULE_NAME}/TRDGeometry.h
  include/${MODULE_NAME}/TRDSimParam.h
  include/${MODULE_NAME}/TRDCommonParam.h
)

SET(LINKDEF src/TRDBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_base_bucket)

O2_GENERATE_LIBRARY()
