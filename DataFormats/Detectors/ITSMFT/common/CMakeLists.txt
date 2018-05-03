set(MODULE_NAME "DataFormatsITSMFT")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Cluster.cxx
  src/ClusterTopology.cxx
  src/TopologyDictionary.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/Cluster.h
  include/${MODULE_NAME}/ClusterTopology.h
  include/${MODULE_NAME}/TopologyDictionary.h
)

Set(LINKDEF src/ITSMFTDataFormatsLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_itsmft_bucket)

O2_GENERATE_LIBRARY()


