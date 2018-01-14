set(MODULE_NAME "CommonUtils")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/TreeStream.cxx
  src/TreeStreamRedirector.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/TreeStream.h
  include/${MODULE_NAME}/TreeStreamRedirector.h
)

Set(LINKDEF src/CommonUtilsLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME common_utils_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testTreeStream.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
