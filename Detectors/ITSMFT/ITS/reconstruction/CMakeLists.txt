set(MODULE_NAME "ITSReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/CookedTrackerTask.cxx
    src/ClustererTask.cxx
    )
#    src/TrivialClustererTask.cxx

set(NO_DICT_SRCS # sources not for the dictionary
    src/CookedTracker.cxx
    src/TrivialVertexer.cxx
    )
#   src/TrivialClusterer.cxx

  
# Headers from sources
string(REPLACE ".cxx" ".h" HEADERS "${SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" HEADERS "${HEADERS}")
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")

Set(LINKDEF src/ITSReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_reconstruction_bucket)
O2_GENERATE_LIBRARY()

