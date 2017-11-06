set(MODULE_NAME "TPCCalibration")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
  src/CalibRawBase.cxx
  src/CalibPedestal.cxx
  src/CalibPulser.cxx
  src/CalibTreeDump.cxx
)

set(HEADERS
  include/${MODULE_NAME}/CalibRawBase.h
  include/${MODULE_NAME}/CalibPedestal.h
  include/${MODULE_NAME}/CalibPulser.h
  include/${MODULE_NAME}/CalibTreeDump.h
)
Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_calibration_bucket)

O2_GENERATE_LIBRARY()
