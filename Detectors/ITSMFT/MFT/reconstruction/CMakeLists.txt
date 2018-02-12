set(MODULE_NAME "MFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ClustererTask.cxx
)
set(HEADERS
  include/${MODULE_NAME}/ClustererTask.h
)


Set(LINKDEF src/MFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME mft_reconstruction_bucket)

O2_GENERATE_LIBRARY()

