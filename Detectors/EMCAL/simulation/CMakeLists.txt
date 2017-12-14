SET(MODULE_NAME EMCALSimulation)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/Digitizer.cxx
  src/DigitizerTask.cxx
  src/SpaceFrame.cxx
)

set(HEADERS
  include/EMCALSimulation/Detector.h
  include/EMCALSimulation/Digitizer.h
  include/EMCALSimulation/DigitizerTask.h
  include/EMCALSimulation/SpaceFrame.h
)

SET(LINKDEF src/EMCALSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_simulation_bucket)

O2_GENERATE_LIBRARY()
