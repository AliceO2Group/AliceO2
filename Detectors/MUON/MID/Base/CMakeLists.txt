set(MODULE_NAME "MIDBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Constants.cxx
  src/GeometryTransformer.cxx
  src/HitFinder.cxx
  src/LegacyUtility.cxx
  src/Mapping.cxx
  src/MpArea.cxx
)

set(NO_DICT_HEADERS
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/GeometryTransformer.h
  include/${MODULE_NAME}/HitFinder.h
  include/${MODULE_NAME}/LegacyUtility.h
  include/${MODULE_NAME}/Mapping.h
  include/${MODULE_NAME}/MpArea.h
  include/${MODULE_NAME}/Serializer.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_base_bucket)

O2_GENERATE_LIBRARY()

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS test/testMapping.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS test/testDEconversion.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS test/testGeometryTransformer.cxx
)
