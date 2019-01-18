set(MODULE_NAME "MCHBase")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/Mapping.cxx
   src/DigitBlock.cxx
   src/PreClusterBlock.cxx
)

set(HEADERS
   include/${MODULE_NAME}/Mapping.h
   include/${MODULE_NAME}/DigitBlock.h
   include/${MODULE_NAME}/PreClusterBlock.h
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_base_bucket)

O2_GENERATE_LIBRARY()

