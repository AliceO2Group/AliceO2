set(MODULE_NAME "MIDBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Mapping.cxx
  src/MpArea.cxx
)

set(NO_DICT_HEADERS
  include/${MODULE_NAME}/Mapping.h
  include/${MODULE_NAME}/MpArea.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_base_bucket)

O2_GENERATE_LIBRARY()

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS test/testMapping.cxx
)
