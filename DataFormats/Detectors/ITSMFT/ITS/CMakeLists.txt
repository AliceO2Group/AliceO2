set(MODULE_NAME "DataFormatsITS")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/TrackITS.cxx
)
set(HEADERS
  include/${MODULE_NAME}/TrackITS.h
)

Set(LINKDEF src/DataFormatsITSLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME data_format_its_bucket)
O2_GENERATE_LIBRARY()

