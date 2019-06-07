if(ENABLE_CUDA)
  add_subdirectory(cuda)
endif()

o2_add_library(ITStracking
               SOURCES
               src/ClusterLines.cxx
               src/Cluster.cxx
               src/ROframe.cxx
               src/IOUtils.cxx
               src/Label.cxx
               src/PrimaryVertexContext.cxx
               src/Road.cxx
               src/Tracker.cxx
               src/TrackerTraitsCPU.cxx
               src/ClusterLines.cxx
               src/Vertexer.cxx
               src/VertexerTraits.cxx
               PUBLIC_LINK_LIBRARIES 
               O2::CommonGPU
               ms_gsl
               O2::CommonConstants
               O2::DataFormatsITSMFT
               O2::SimulationDataFormat
               O2::ITSBase
               O2::DataFormatsITS)


o2_target_root_dictionary(ITStracking
                          HEADERS
                          include/ITStracking/ClusterLines.h
                          include/ITStracking/Tracklet.h
                          LINKDEF
                          src/TrackingLinkDef.h)
