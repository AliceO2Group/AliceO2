# @author Mikolaj Krzewicki

set(MODULE_NAME "O2MessageMonitor")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/O2MessageMonitor.cxx
)

set(HEADERS
  include/${MODULE_NAME}/O2MessageMonitor.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2MessageMonitor_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2MessageMonitor_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
  EXE_NAME runO2MessageMonitor
  SOURCES src/runO2MessageMonitor.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS()

set(TEST_SRCS
  test/O2MessageMonitorTest.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
