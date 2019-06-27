# @author David Rohr
# @brief  TPC data formats

set(MODULE_NAME "DataFormatsTPC")
set(MODULE_BUCKET_NAME data_format_TPC_bucket)

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Helpers.cxx
   src/TrackTPC.cxx
   src/TPCSectorHeader.cxx
   src/ClusterNativeHelper.cxx
)

set(HEADERS
   include/${MODULE_NAME}/ClusterGroupAttribute.h
   include/${MODULE_NAME}/ClusterNative.h
   include/${MODULE_NAME}/ClusterNativeHelper.h
   include/${MODULE_NAME}/ClusterHardware.h
   include/${MODULE_NAME}/Helpers.h
   include/${MODULE_NAME}/TrackTPC.h
   include/${MODULE_NAME}/Constants.h
   include/${MODULE_NAME}/Defs.h
   include/${MODULE_NAME}/dEdxInfo.h
   include/${MODULE_NAME}/CompressedClusters.h
)

set(LINKDEF src/DataFormatsTPCLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testClusterNative.cxx
  test/testClusterHardware.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
