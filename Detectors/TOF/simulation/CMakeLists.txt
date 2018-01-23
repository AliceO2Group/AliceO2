SET(MODULE_NAME TOFSimulation)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/Digitizer.cxx
  src/DigitizerTask.cxx
)

set(HEADERS
  include/TOFSimulation/Detector.h
  include/TOFSimulation/Digitizer.h
  include/TOFSimulation/DigitizerTask.h
)

SET(LINKDEF src/TOFSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME tof_simulation_bucket)

O2_GENERATE_LIBRARY()
