set(MODULE_NAME "ITSMFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Cluster.cxx
  src/PixelReader.cxx
  src/Clusterer.cxx
)
set(HEADERS
  include/${MODULE_NAME}/Cluster.h
  include/${MODULE_NAME}/PixelReader.h
  include/${MODULE_NAME}/Clusterer.h
)
Set(LINKDEF src/ITSMFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME itsmft_reconstruction_bucket)
O2_GENERATE_LIBRARY()

