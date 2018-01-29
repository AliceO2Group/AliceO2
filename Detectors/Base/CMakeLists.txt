set(MODULE_NAME "DetectorsBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/GeometryManager.cxx
  src/MaterialManager.cxx
  src/Propagator.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/GeometryManager.h
  include/${MODULE_NAME}/MaterialManager.h
  include/${MODULE_NAME}/Propagator.h
)

Set(LINKDEF src/DetectorsBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME detectors_base_bucket)

O2_GENERATE_LIBRARY()
