set(MODULE_NAME "ZDCSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Detector.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Detector.h
    )

Set(LINKDEF src/ZDCSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME zdc_simulation_bucket)

O2_GENERATE_LIBRARY()
