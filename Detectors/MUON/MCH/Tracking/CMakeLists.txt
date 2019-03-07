set(MODULE_NAME "MCHTracking")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Cluster.cxx
   src/TrackParam.cxx
   src/Track.cxx
   src/TrackExtrap.cxx
   src/TrackFitter.cxx
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_tracking_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_tracking_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
  EXE_NAME trackfitter-workflow

  SOURCES
  src/TrackFitterWorkflow.cxx
  src/TrackSamplerSpec.cxx
  src/TrackFitterSpec.cxx
  src/TrackSinkSpec.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}

  BUCKET_NAME ${BUCKET_NAME}
)
