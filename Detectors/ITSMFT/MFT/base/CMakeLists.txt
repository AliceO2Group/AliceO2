set(MODULE_NAME "MFTBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Constants.cxx
  src/EventHeader.cxx
)

set(HEADERS
  include/${MODULE_NAME}/Constants.h
  include/${MODULE_NAME}/EventHeader.h
)

Set(LINKDEF src/MFTBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mft_base_bucket)

O2_GENERATE_LIBRARY()

