set(MODULE_NAME "TPCsimulation")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Detector.cxx
   src/Point.cxx

)

set(HEADERS
    include/${MODULE_NAME}/Detector.h
    include/${MODULE_NAME}/Point.h
)
Set(LINKDEF src/tpcSimulationLinkDef.h)
Set(LIBRARY_NAME tpcSimulation)
set(BUCKET_NAME tpc_simulation_bucket)

O2_GENERATE_LIBRARY()
