set(MODULE_NAME "T0Simulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  #src/DigitizerTask.cxx
  )

set(HEADERS
    include/${MODULE_NAME}/Detector.h
    #include/${MODULE_NAME}/DigitizerTask.h
    )

Set(LINKDEF src/T0SimulationLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_simulation_bucket)

O2_GENERATE_LIBRARY()
