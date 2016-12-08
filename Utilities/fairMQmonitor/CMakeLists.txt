# @author Mikolaj Krzewicki

set(MODULE_NAME "fairMQmonitor")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/FairMQmonitor.cxx
)

set(HEADERS
  include/${MODULE_NAME}/FairMQmonitor.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fairMQmonitor_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fairMQmonitor_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
  EXE_NAME runFairMQmonitor
  SOURCES src/runFairMQmonitor.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS()

set(TEST_SRCS
  test/fairMQmonitorTest.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
