SET(MODULE_NAME TRDBase)

O2_SETUP(NAME ${MODULE_NAME})

SET(SRCS
  src/TRDPadPlane.cxx
  src/TRDGeometryBase.cxx
  src/TRDGeometry.cxx
  src/TRDGeometryFlat.cxx
  src/TRDCommonParam.cxx
  src/TRDSimParam.cxx
  src/Digit.cxx
  src/TRDArrayADC.cxx
  src/TRDArrayDictionary.cxx
  src/TRDArraySignal.cxx
  src/TRDCalPadStatus.cxx
  src/TRDCalSingleChamberStatus.cxx
  src/TRDCalDet.cxx
  src/TRDCalROC.cxx
  src/TRDFeeParam.cxx
  src/TRDSignalIndex.cxx
)

SET(HEADERS
  include/${MODULE_NAME}/TRDPadPlane.h
  include/${MODULE_NAME}/TRDGeometryBase.h
  include/${MODULE_NAME}/TRDGeometry.h
  include/${MODULE_NAME}/TRDGeometryFlat.h
  include/${MODULE_NAME}/TRDSimParam.h
  include/${MODULE_NAME}/TRDCommonParam.h
  include/${MODULE_NAME}/Digit.h
  include/${MODULE_NAME}/DigitIndex.h
  include/${MODULE_NAME}/TRDDigitsParam.h
  include/${MODULE_NAME}/TRDArrayADC.h
  include/${MODULE_NAME}/TRDArrayDictionary.h
  include/${MODULE_NAME}/TRDArraySignal.h
  include/${MODULE_NAME}/TRDCalPadStatus.h
  include/${MODULE_NAME}/TRDCalSingleChamberStatus.h
  include/${MODULE_NAME}/TRDCalDet.h
  include/${MODULE_NAME}/TRDCalROC.h
  include/${MODULE_NAME}/TRDFeeParam.h
  include/${MODULE_NAME}/TRDSignalIndex.h
)

SET(LINKDEF src/TRDBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME trd_base_bucket)

O2_GENERATE_LIBRARY()
