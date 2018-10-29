set(MODULE_NAME "ITStracking")

O2_SETUP(NAME ${MODULE_NAME})

set(NO_DICT_SRCS # sources not for the dictionary
    src/Cell.cxx
    src/Cluster.cxx
    src/ROframe.cxx
    src/IOUtils.cxx
    src/Label.cxx
    src/PrimaryVertexContext.cxx
    src/Road.cxx
    src/Tracker.cxx
    src/TrackerTraits.cxx
    src/Tracklet.cxx
    src/ClusterLines.cxx
    src/Vertexer.cxx
    )
#   src/TrivialClusterer.cxx


# Headers from sources
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")

set(NO_DICT_HEADERS
    ${NO_DICT_HEADERS}
    include/${MODULE_NAME}/Config.h
    include/${MODULE_NAME}/Definitions.h
    )

# Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME its_tracking_bucket)
O2_GENERATE_LIBRARY()
