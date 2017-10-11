# @author Gvozden Neskovic
# @brief  DataDistribution and DPL integration

o2_define_bucket(
    NAME
    SubTimeFrameBuilderDplAdapter_bucket

    DEPENDENCIES
    DataDistributionBase_bucket
    O2Device_bucket
    FrameworkApplication_bucket
)

set(MODULE_NAME "SubTimeFrameBuilderDplAdapter")
set(MODULE_BUCKET_NAME SubTimeFrameBuilderDplAdapter_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# Tests
set(EXE_SRC
    test_StfBuilderDplAdapter.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "StfBuilderDplAdapter"
    SOURCES ${EXE_SRC}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/start_StfBuilderDplAdapterTest.sh.in
      ${CMAKE_BINARY_DIR}/bin/start_StfBuilderDplAdapterTest.sh
)

install(
  PROGRAMS ${CMAKE_BINARY_DIR}/bin/start_StfBuilderDplAdapterTest.sh
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
