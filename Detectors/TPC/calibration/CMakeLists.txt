set(MODULE_NAME "TPCCalibration")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
  src/CalibPedestal.cxx
  src/CalibRawBase.cxx
)

set(HEADERS
  include/${MODULE_NAME}/CalibPedestal.h
  include/${MODULE_NAME}/CalibRawBase.h
)
Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_calibration_bucket)

O2_GENERATE_LIBRARY()
