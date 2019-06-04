set(MODULE_NAME "ITSSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/V11Geometry.cxx
    src/V1Layer.cxx
    src/V3Layer.cxx
    src/V3Services.cxx
    src/Detector.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Detector.h
    include/${MODULE_NAME}/V1Layer.h
    include/${MODULE_NAME}/V3Layer.h
    include/${MODULE_NAME}/V3Services.h
    include/${MODULE_NAME}/V11Geometry.h
    )

Set(LINKDEF src/ITSSimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME its_simulation_bucket)

O2_GENERATE_LIBRARY()
