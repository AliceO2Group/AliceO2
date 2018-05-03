set(MODULE_NAME "DataFormatsMFT")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/TrackMFT.cxx
)
set(HEADERS
  include/${MODULE_NAME}/TrackMFT.h
)

Set(LINKDEF src/DataFormatsMFTLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME data_format_mft_bucket)
O2_GENERATE_LIBRARY()

