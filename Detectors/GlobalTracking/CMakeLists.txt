set(MODULE_NAME "GlobalTracking")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/MatchTPCITS.cxx
   src/MatchTOF.cxx
)

set(HEADERS
   include/${MODULE_NAME}/MatchTPCITS.h
   include/${MODULE_NAME}/MatchTOF.h
)

set(LINKDEF src/GlobalTrackingLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME global_tracking_bucket)

O2_GENERATE_LIBRARY()

