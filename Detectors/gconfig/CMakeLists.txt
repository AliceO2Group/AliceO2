set(MODULE_NAME "SimSetup")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
     src/G3Config.cxx
     src/G4Config.cxx
     src/SimSetup.cxx
   )

set(HEADERS
    include/${MODULE_NAME}/SimSetup.h
   )

Set(LINKDEF src/GConfLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME simulation_setup_bucket)

O2_GENERATE_LIBRARY()
