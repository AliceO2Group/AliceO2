set(MODULE_NAME "FDDSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Detector.h
    include/${MODULE_NAME}/Hit.h
    )

Set(LINKDEF src/FDDSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_simulation_bucket)

O2_GENERATE_LIBRARY()
