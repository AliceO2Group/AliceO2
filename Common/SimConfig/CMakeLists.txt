set(MODULE_NAME "SimConfig")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/SimConfig.cxx
   )

set(HEADERS
    include/${MODULE_NAME}/SimConfig.h
   )

set(LINKDEF src/SimConfigLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME configuration_bucket)

O2_GENERATE_LIBRARY()

 O2_GENERATE_EXECUTABLE(
      EXE_NAME testSimConf
      SOURCES test/TestConfig.cxx
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )

