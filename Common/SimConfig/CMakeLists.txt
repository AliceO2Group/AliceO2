set(MODULE_NAME "SimConfig")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/SimConfig.cxx
    src/ConfigurableParam.cxx
    src/ConfigurableParamHelper.cxx
    src/SimCutParams.cxx
    src/G4Params.cxx
   )

set(HEADERS
    include/${MODULE_NAME}/SimConfig.h
    include/${MODULE_NAME}/SimCutParams.h
    include/${MODULE_NAME}/G4Params.h
    include/${MODULE_NAME}/ConfigurableParam.h
    include/${MODULE_NAME}/ConfigurableParamHelper.h
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

O2_GENERATE_EXECUTABLE(
  EXE_NAME testConfigurableParam
  SOURCES test/TestConfigurableParam.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

