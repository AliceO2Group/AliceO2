SET(MODULE_NAME TRDSimulation)

O2_SETUP(NAME ${ MODULE_NAME })

set(SRCS src / Detector.cxx)

  set(HEADERS include / ${ MODULE_NAME } / Detector.h)

    SET(LINKDEF src / TRDSimulationLinkDef.h) SET(LIBRARY_NAME ${ MODULE_NAME })
#for the moment we can reuse a different bucket
      SET(BUCKET_NAME trd_simulation_bucket)

        O2_GENERATE_LIBRARY()
