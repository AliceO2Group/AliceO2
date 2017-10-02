set(MODULE_NAME "MFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/PixelReader.cxx
  src/ClustererTask.cxx
  src/Track.cxx
  src/FindTracks.cxx
)
set(HEADERS
  include/${MODULE_NAME}/PixelReader.h
  include/${MODULE_NAME}/ClustererTask.h
  include/${MODULE_NAME}/Track.h
  include/${MODULE_NAME}/FindTracks.h
)


Set(LINKDEF src/MFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME mft_reconstruction_bucket)

O2_GENERATE_LIBRARY()

