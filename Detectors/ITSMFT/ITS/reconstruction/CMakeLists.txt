set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Cluster.cxx
    src/TrivialClustererTask.cxx
    src/ClustererTask.cxx
    src/CookedTrack.cxx
    src/CookedTrackerTask.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Cluster.h
    include/${MODULE_NAME}/TrivialClustererTask.h
    include/${MODULE_NAME}/ClustererTask.h
    include/${MODULE_NAME}/CookedTrack.h
    include/${MODULE_NAME}/CookedTrackerTask.h
    )
set(NO_DICT_SRCS # sources not for the dictionary
    src/TrivialClusterer.cxx
    src/Clusterer.cxx
    src/CAaux.cxx
    src/CATracker.cxx
    src/CATrackingStation.cxx
    src/CookedTracker.cxx
    )
set(NO_DICT_HEADERS # sources not for the dictionary
    include/${MODULE_NAME}/TrivialClusterer.h
    include/${MODULE_NAME}/Clusterer.h
    include/${MODULE_NAME}/CAaux.h
    include/${MODULE_NAME}/CATracker.h
    include/${MODULE_NAME}/CATrackingStation.h
    include/${MODULE_NAME}/CookedTracker.h
    )
Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)
O2_GENERATE_LIBRARY()

