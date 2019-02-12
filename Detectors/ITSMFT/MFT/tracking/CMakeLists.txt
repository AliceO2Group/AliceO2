set(MODULE_NAME "MFTTracking")

O2_SETUP(NAME ${MODULE_NAME})

set(NO_DICT_SRCS # sources not for the dictionary
    src/Tracker.cxx
    src/ROframe.cxx
    src/Cluster.cxx
    src/IOUtils.cxx
    )

# Headers from sources
string(REPLACE ".cxx" ".h" NO_DICT_HEADERS "${NO_DICT_SRCS}")
string(REPLACE "src" "include/${MODULE_NAME}" NO_DICT_HEADERS "${NO_DICT_HEADERS}")

#set(NO_DICT_HEADERS
#    ${NO_DICT_HEADERS}
#    )

set(HEADERS
  include/${MODULE_NAME}/TrackCA.h
)

Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME mft_tracking_bucket)
O2_GENERATE_LIBRARY()

