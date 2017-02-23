SET(MODULE_NAME EMCALSimulation)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
)

set(HEADERS
  include/EMCALSimulation/Detector.h
)

SET(LINKDEF src/EMCALSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_simulation_bucket)

O2_GENERATE_LIBRARY()
