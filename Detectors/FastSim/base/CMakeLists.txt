SET(MODULE_NAME FastSimBase)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/DirectTransport.cxx
  src/DirectTransportBuilder.cxx
)

set(HEADERS
  include/${MODULE_NAME}/DirectTransport.h
  include/${MODULE_NAME}/DirectTransportBuilder.h
)

#SET(LINKDEF src/FastSimBaseLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME fastsim_base_bucket)

O2_GENERATE_LIBRARY()
