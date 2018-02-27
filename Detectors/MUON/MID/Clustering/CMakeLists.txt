set(MODULE_NAME "MIDClustering")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Clusterizer.cxx
   src/ClusterizerDevice.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_clustering_bucket)

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME runMIDClusterizer
  SOURCES src/runMIDClusterizer.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config/runMIDClusterizer.json DESTINATION etc/config)

add_subdirectory(test)
