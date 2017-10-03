set(MODULE_NAME "SimulationDataFormat")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Stack.cxx
  src/MCTrack.cxx
  src/MCCompLabel.cxx
  src/MCEventHeader.cxx
  src/GeneratorHeader.cxx
  src/GeneratorInfo.cxx
  src/HeavyIonInfo.cxx
  src/CrossSectionInfo.cxx
)

Set(HEADERS
    include/${MODULE_NAME}/Stack.h
    include/${MODULE_NAME}/MCTrack.h
    include/${MODULE_NAME}/BaseHits.h
    include/${MODULE_NAME}/MCTruthContainer.h
    include/${MODULE_NAME}/MCCompLabel.h
    include/${MODULE_NAME}/MCInteractionRecord.h
    include/${MODULE_NAME}/TrackReference.h
    include/${MODULE_NAME}/MCEventHeader.h
    include/${MODULE_NAME}/GeneratorHeader.h
    include/${MODULE_NAME}/GeneratorInfo.h
    include/${MODULE_NAME}/HeavyIonInfo.h
    include/${MODULE_NAME}/CrossSectionInfo.h
)

Set(LINKDEF src/SimulationDataLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_simulation_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testBasicHits.cxx
  test/testMCTruthContainer.cxx
  test/testMCCompLabel.cxx
  test/MCTrack.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
