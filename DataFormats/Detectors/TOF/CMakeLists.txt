set(MODULE_NAME "DataFormatsTOF")
set(MODULE_BUCKET_NAME data_format_TOF_bucket)

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Cluster.cxx
)

set(HEADERS
   include/${MODULE_NAME}/Cluster.h
)

set(LINKDEF src/DataFormatsTOFLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

