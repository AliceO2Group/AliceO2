set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/TrivialClustererTask.cxx
    src/ClustererTask.cxx
    src/CookedTrackerTask.cxx
    src/CA/TrackerTask.cxx
    )

set(NO_DICT_SRCS # sources not for the dictionary
    src/TrivialClusterer.cxx
    src/TrivialVertexer.cxx
    src/CookedTracker.cxx
    src/CA/Cell.cxx
    src/CA/Cluster.cxx
    src/CA/Event.cxx
    src/CA/IOUtils.cxx
    src/CA/Label.cxx
    src/CA/Layer.cxx
    src/CA/PrimaryVertexContext.cxx
    src/CA/Road.cxx
    src/CA/Tracker.cxx
    src/CA/TrackingUtils.cxx
    src/CA/Tracklet.cxx
    )

# Headers from sources
string(REPLACE ".cxx" ".h" HEADERS "${SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" HEADERS "${HEADERS}")
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")

# Headers without corresponding sources
set(HEADERS
  include/${MODULE_NAME}/CA/Track.h
  ${HEADERS}
  )

set(NO_DICT_HEADERS # sources not for the dictionary
    include/${MODULE_NAME}/TrivialClusterer.h
    include/${MODULE_NAME}/CookedTracker.h
    )

Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)
O2_GENERATE_LIBRARY()

