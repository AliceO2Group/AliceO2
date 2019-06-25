SET(MODULE_NAME EMCALCalib)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/BadChannelMap.cxx
)

set(HEADERS
  include/${MODULE_NAME}/BadChannelMap.h
)

SET(LINKDEF src/EMCALCalibLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME emcal_calib_bucket)

O2_GENERATE_LIBRARY()
