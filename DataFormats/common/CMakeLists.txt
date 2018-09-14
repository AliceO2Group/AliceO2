set(MODULE_NAME "CommonDataFormat")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/InteractionRecord.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/TimeStamp.h
  include/${MODULE_NAME}/EvIndex.h
  include/${MODULE_NAME}/InteractionRecord.h
)

Set(LINKDEF src/CommonDataFormatLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_common_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testTimeStamp.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
