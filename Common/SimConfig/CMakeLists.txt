o2_add_library(SimConfig
               SOURCES
               src/SimConfig.cxx
               src/ConfigurableParam.cxx
               src/ConfigurableParamHelper.cxx
               src/SimCutParams.cxx
               PUBLIC_LINK_LIBRARIES
               O2::CommonUtils
               O2::DetectorsCommonDataFormats
               FairRoot::Base
               Boost::program_options
               Boost::filesystem)

o2_target_root_dictionary(SimConfig
                          HEADERS
                          include/SimConfig/SimConfig.h
                          include/SimConfig/SimCutParams.h
                          include/SimConfig/ConfigurableParam.h
                          include/SimConfig/ConfigurableParamHelper.h)
#
# O2_GENERATE_EXECUTABLE( EXE_NAME testSimConf SOURCES test/TestConfig.cxx
# MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME ${BUCKET_NAME} )
#
# O2_GENERATE_EXECUTABLE( EXE_NAME testConfigurableParam SOURCES
# test/TestConfigurableParam.cxx MODULE_LIBRARY_NAME ${LIBRARY_NAME} BUCKET_NAME
# ${BUCKET_NAME} )
#
