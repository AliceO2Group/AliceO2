# @author Mikolaj Krzewicki

set(MODULE_NAME "ShmSource")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/ShmSource.cxx
  src/ShmSink.cxx
)

set(HEADERS
  include/${MODULE_NAME}/ShmSource.h
  include/${MODULE_NAME}/ShmSink.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2device_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2device_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
  EXE_NAME runShmSource
  SOURCES src/runShmSource.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME runShmSink
  SOURCES src/runShmSink.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS()
#
#set(TEST_SRCS
#  test/ShmSourceTest.cxx
#)
#
#O2_GENERATE_TESTS(
#  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
#  BUCKET_NAME ${BUCKET_NAME}
#  TEST_SRCS ${TEST_SRCS}
#)
