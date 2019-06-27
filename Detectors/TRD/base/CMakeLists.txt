SET(MODULE_NAME TRDBase)

O2_SETUP(NAME ${MODULE_NAME})

SET(SRCS
  src/TRDPadPlane.cxx
  src/TRDGeometryBase.cxx
  src/TRDGeometry.cxx
  src/TRDGeometryFlat.cxx
  src/TRDCommonParam.cxx
  src/TRDSimParam.cxx
  src/PadResponse.cxx
  src/Digit.cxx
  src/TRDCalPadStatus.cxx
  src/TRDCalSingleChamberStatus.cxx
  src/CalDet.cxx
  src/CalROC.cxx
  src/TRDFeeParam.cxx
)

SET(HEADERS
  include/${MODULE_NAME}/TRDPadPlane.h
  include/${MODULE_NAME}/TRDGeometryBase.h
  include/${MODULE_NAME}/TRDGeometry.h
  include/${MODULE_NAME}/TRDGeometryFlat.h
  include/${MODULE_NAME}/TRDSimParam.h
  include/${MODULE_NAME}/TRDCommonParam.h
  include/${MODULE_NAME}/PadResponse.h
  include/${MODULE_NAME}/Digit.h
  include/${MODULE_NAME}/MCLabel.h
  include/${MODULE_NAME}/TRDCalPadStatus.h
  include/${MODULE_NAME}/TRDCalSingleChamberStatus.h
  include/${MODULE_NAME}/CalDet.h
  include/${MODULE_NAME}/CalROC.h
  include/${MODULE_NAME}/TRDFeeParam.h
)

SET(LINKDEF src/TRDBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME trd_base_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
   test/testTRDDiffusionCoefficient.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
