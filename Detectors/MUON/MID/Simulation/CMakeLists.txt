set(MODULE_NAME "MIDSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Detector.cxx
   src/Materials.cxx
   src/Materials.h
   src/Geometry.cxx
   src/Geometry.h
)

set(HEADERS
   include/${MODULE_NAME}/Detector.h
)

SET(LINKDEF src/MIDSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_simulation_bucket)

O2_GENERATE_LIBRARY()
