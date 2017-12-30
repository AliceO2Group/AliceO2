SET(MODULE_NAME PHOSSimulation)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/GeometryParams.cxx
)

set(HEADERS
  include/PHOSSimulation/Detector.h
  include/PHOSSimulation/GeometryParams.h
)

SET(LINKDEF src/PHOSSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME phos_simulation_bucket)

O2_GENERATE_LIBRARY()
