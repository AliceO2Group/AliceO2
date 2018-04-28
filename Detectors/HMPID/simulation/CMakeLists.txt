set(MODULE_NAME "HMPIDSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Detector.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Detector.h
    )

Set(LINKDEF src/HMPIDSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME hmpid_simulation_bucket)

O2_GENERATE_LIBRARY()
