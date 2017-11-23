# @author Mikolaj Krzewicki

set(MODULE_NAME "Headers")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/DataHeader.cxx
  src/NameHeader.cxx
  src/HeartbeatFrame.cxx
  src/TimeStamp.cxx
)

set(HEADERS
  include/${MODULE_NAME}/DataHeader.h
  include/${MODULE_NAME}/NameHeader.h
  include/${MODULE_NAME}/HeartbeatFrame.h
  include/${MODULE_NAME}/TimeStamp.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fairroot_base_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testDataHeader.cxx
  test/testTimeStamp.cxx
  test/test_HeartbeatFrame.cxx
  test/test_RAWDataHeader.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
