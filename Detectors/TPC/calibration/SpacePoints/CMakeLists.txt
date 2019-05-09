SET(MODULE_NAME SpacePoints)

O2_SETUP(NAME ${MODULE_NAME})

SET(SRCS
  src/Param.cxx
  src/TrackResiduals.cxx
)

SET(HEADERS
  include/${MODULE_NAME}/Param.h
  include/${MODULE_NAME}/TrackResiduals.h
)

SET(LINKDEF src/SpacePointCalibLinkDef.h)
SET(LIBRARY_NAME ${MODULE_NAME})
SET(BUCKET_NAME spacepoint_calib_bucket)

O2_GENERATE_LIBRARY()
