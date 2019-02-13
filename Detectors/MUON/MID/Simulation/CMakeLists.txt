set(MODULE_NAME "MIDSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/Geometry.cxx
  src/Hit.cxx
  src/Materials.cxx
  src/Stepper.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/Hit.h
  include/${MODULE_NAME}/Stepper.h
)

set(LINKDEF src/MIDSimulationLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_simulation_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)