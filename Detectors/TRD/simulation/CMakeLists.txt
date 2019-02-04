SET(MODULE_NAME TRDSimulation)

O2_SETUP(NAME ${MODULE_NAME})

SET(SRCS
  src/Detector.cxx
  src/TRsim.cxx
  src/Digitizer.cxx
)
SET(HEADERS
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/TRsim.h
  include/${MODULE_NAME}/Digitizer.h
)

SET(LINKDEF src/TRDSimulationLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME trd_simulation_bucket)

O2_GENERATE_LIBRARY()
