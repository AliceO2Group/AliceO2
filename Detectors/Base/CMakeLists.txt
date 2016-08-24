set(MODULE_NAME "DetectorsBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/TrackReference.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/TrackReference.h
  include/${MODULE_NAME}/Detector.h
)
Set(LINKDEF src/BaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME detectors_base)

O2_GENERATE_LIBRARY()
