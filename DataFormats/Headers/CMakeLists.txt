# @author Mikolaj Krzewicki

set(MODULE_NAME "Headers")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/DataHeader.cxx
)

set(HEADERS
  include/${MODULE_NAME}/DataHeader.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fairroot_base_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/dataHeaderTest.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
