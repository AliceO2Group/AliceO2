# @author David Rohr
# @brief  TPC data formats

set(MODULE_NAME "DataFormatsTPC")
set(MODULE_BUCKET_NAME DataFormatsTPC_bucket)

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Helpers.cxx
)

set(HEADERS
   include/${MODULE_NAME}/ClusterNative.h
   include/${MODULE_NAME}/ClusterHardware.h
   include/${MODULE_NAME}/Helpers.h
)

set(LINKDEF src/DataFormatsTPCLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()
