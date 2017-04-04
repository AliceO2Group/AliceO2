set(MODULE_NAME "TPCReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/DigitData.cxx
)

set(HEADERS
   include/${MODULE_NAME}/DigitData.h
)
Set(LINKDEF src/TPCReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_reconstruction_bucket)

O2_GENERATE_LIBRARY()
