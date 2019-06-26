if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

set(MODULE_NAME "ITStracking")

O2_SETUP(NAME ${MODULE_NAME})

#DEBUG
set(SRC
    src/ClusterLines.cxx
)

set(HEADERS
    include/${MODULE_NAME}/ClusterLines.h
    include/${MODULE_NAME}/Tracklet.h
    include/${MODULE_NAME}/DBScan.h
)

Set(LINKDEF src/TrackingLinkDef.h)
#\DEBUG

set(NO_DICT_SRCS # sources not for the dictionary
    src/Cluster.cxx
    src/ROframe.cxx
    src/Graph.cxx
    src/DBScan.cxx
    src/IOUtils.cxx
    src/Label.cxx
    src/PrimaryVertexContext.cxx
    src/Road.cxx
    src/Tracker.cxx
    src/TrackerTraitsCPU.cxx
    src/ClusterLines.cxx
    src/Vertexer.cxx
    src/VertexerTraits.cxx
    )
#   src/TrivialClusterer.cxx


# Headers from sources
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")

set(NO_DICT_HEADERS
    ${NO_DICT_HEADERS}
    include/${MODULE_NAME}/Cell.h
    include/${MODULE_NAME}/Config.h
    include/${MODULE_NAME}/Definitions.h
    include/${MODULE_NAME}/TrackerTraits.h
    include/${MODULE_NAME}/Tracklet.h
    )

# Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_tracking_bucket)
O2_GENERATE_LIBRARY()

