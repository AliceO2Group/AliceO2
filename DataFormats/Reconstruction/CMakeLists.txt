set(MODULE_NAME "ReconstructionDataFormats")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Track.cxx
  src/BaseCluster.cxx
  src/TrackTPCITS.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/Track.h
  include/${MODULE_NAME}/BaseCluster.h
  include/${MODULE_NAME}/TrackTPCITS.h
)

Set(LINKDEF src/ReconstructionDataFormatsLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_reconstruction_bucket)

O2_GENERATE_LIBRARY()


