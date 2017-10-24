set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/TrivialClustererTask.cxx
    src/ClustererTask.cxx
    src/CookedTrack.cxx
    src/CookedTrackerTask.cxx
    src/CA/Track.cxx
    src/CA/TrackerTask.cxx
    )

set(NO_DICT_SRCS
    src/TrivialClusterer.cxx
    src/CookedTracker.cxx
    src/CA/Cell.cxx
    src/CA/Cluster.cxx
    src/CA/Event.cxx
    src/CA/Layer.cxx
    src/CA/PrimaryVertexContext.cxx
    src/CA/Road.cxx
    src/CA/Tracker.cxx
    src/CA/TrackingUtils.cxx
    src/CA/Tracklet.cxx
    )

# Headers from sources
string(REPLACE ".cxx" ".h" HEADERS_WRONG_DIR "${SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" HEADERS "${HEADERS_WRONG_DIR}")
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS_WRONG_DIR "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS_WRONG_DIR}")

Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)
O2_GENERATE_LIBRARY()

