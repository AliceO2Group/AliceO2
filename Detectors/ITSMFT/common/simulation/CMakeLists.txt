set(MODULE_NAME "ITSMFTSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Hit.cxx
    src/ClusterShape.cxx
    src/AlpideSimResponse.cxx
    src/Chip.cxx
    src/AlpideChip.cxx
    src/SimuClusterShaper.cxx
    src/SimulationAlpide.cxx
    src/DigiParams.cxx
    src/Digitizer.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Hit.h
    include/${MODULE_NAME}/ClusterShape.h
    include/${MODULE_NAME}/AlpideSimResponse.h
    include/${MODULE_NAME}/DigiParams.h
    include/${MODULE_NAME}/Digitizer.h
    include/${MODULE_NAME}/Chip.h
    include/${MODULE_NAME}/AlpideChip.h
    include/${MODULE_NAME}/SimuClusterShaper.h
    include/${MODULE_NAME}/SimulationAlpide.h
    )

Set(LINKDEF src/ITSMFTSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME itsmft_simulation_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testAlpideSimResponse.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
