set(MODULE_NAME "MFTSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/DigitizerTask.cxx
)
set(HEADERS
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/DigitizerTask.h
)

Set(LINKDEF src/MFTSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mft_simulation_bucket)

O2_GENERATE_LIBRARY()

