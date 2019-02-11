set(MODULE_NAME "MIDClustering")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Clusterizer.cxx
   src/PreClusterizer.cxx
   src/PreClusters.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_clustering_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)
