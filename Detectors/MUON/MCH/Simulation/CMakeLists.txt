set(MODULE_NAME "MCHSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Detector.cxx
   src/Geometry.cxx
   src/GeometryTest.cxx
   src/Hit.cxx
   src/Materials.cxx
   src/Materials.h
   src/Station1Geometry.cxx
   src/Station1Geometry.h
   src/Station2Geometry.cxx
   src/Station2Geometry.h
   src/Station345Geometry.cxx
   src/Station345Geometry.h
   src/Stepper.cxx
   src/Stepper.h
)

set(HEADERS
   include/${MODULE_NAME}/Detector.h
   include/${MODULE_NAME}/Geometry.h
   include/${MODULE_NAME}/GeometryTest.h
   include/${MODULE_NAME}/Hit.h
)

SET(LINKDEF src/MCHSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_simulation_bucket)

O2_GENERATE_LIBRARY()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()