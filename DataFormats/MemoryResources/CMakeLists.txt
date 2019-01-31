# @author Mikolaj Krzewicki

set(MODULE_NAME "MemoryResources")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/MemoryResources.cxx
)

set(HEADERS
  include/${MODULE_NAME}/MemoryResources.h
  include/${MODULE_NAME}/observer_ptr.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME pmr_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testMemoryResources.cxx
  test/test_observer_ptr.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
