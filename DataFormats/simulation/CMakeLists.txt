set(MODULE_NAME "SimulationDataFormat")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Stack.cxx
  src/MCTrack.cxx
)

Set(HEADERS
    include/${MODULE_NAME}/Stack.h
    include/${MODULE_NAME}/MCTrack.h
    include/${MODULE_NAME}/BaseHits.h
)

Set(LINKDEF src/SimulationDataLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_simulation_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testBasicHits.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
