set(MODULE_NAME "DetectorsCommonDataFormats")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/DetID.cxx
  src/AlignParam.cxx
  src/DetMatrixCache.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/DetID.h
  include/${MODULE_NAME}/AlignParam.h
  include/${MODULE_NAME}/DetMatrixCache.h
)

Set(LINKDEF src/DetectorsCommonDataFormatsLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_detectors_common_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testDetID.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)


