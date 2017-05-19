set(MODULE_NAME "DetectorsBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/Track.cxx
  src/TrackReference.cxx
  src/ContVec.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/Track.h
  include/${MODULE_NAME}/TrackReference.h
  include/${MODULE_NAME}/Utils.h
  include/${MODULE_NAME}/ContVec.h
)

Set(LINKDEF src/BaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME detectors_base)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
   test/testContVec.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  MODULE_LIBRARY_NAME ${MODULE_NAME}
  TEST_SRCS ${TEST_SRCS}
)
