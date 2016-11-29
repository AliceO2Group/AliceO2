set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Cluster.cxx
    src/Clusterer.cxx
    src/ClustererTask.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Cluster.h
    include/${MODULE_NAME}/Clusterer.h
    include/${MODULE_NAME}/ClustererTask.h
    )
set(NO_DICT_SRCS # sources not for the dictionary
    src/CAaux.cxx
    src/CATracker.cxx
    src/CATrackingStation.cxx
    )
set(NO_DICT_HEADERS # sources not for the dictionary
    include/${MODULE_NAME}/CAaux.h
    include/${MODULE_NAME}/CATracker.h
    include/${MODULE_NAME}/CATrackingStation.h
    )
Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)
O2_GENERATE_LIBRARY()

