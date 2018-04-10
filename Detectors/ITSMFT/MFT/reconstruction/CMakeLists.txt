set(MODULE_NAME "MFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ClustererTask.cxx
  src/TrackerTask.cxx
)
set(HEADERS
  include/${MODULE_NAME}/ClustererTask.h
  include/${MODULE_NAME}/TrackerTask.h
)
set(NO_DICT_SRCS # sources not for the dictionary
  src/Tracker.cxx
)
set(NO_DICT_HEADERS # sources not for the dictionary
  include/${MODULE_NAME}/Tracker.h
)

Set(LINKDEF src/MFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME mft_reconstruction_bucket)

O2_GENERATE_LIBRARY()

