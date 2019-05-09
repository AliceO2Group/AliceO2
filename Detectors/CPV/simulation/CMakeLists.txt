SET(MODULE_NAME CPVSimulation)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/GeometryParams.cxx
  src/Digitizer.cxx
  src/DigitizerTask.cxx
)

set(HEADERS
  include/CPVSimulation/Detector.h
  include/CPVSimulation/GeometryParams.h
  include/CPVSimulation/Digitizer.h
  include/CPVSimulation/DigitizerTask.h
)

SET(LINKDEF src/CPVSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME cpv_simulation_bucket)

O2_GENERATE_LIBRARY()
