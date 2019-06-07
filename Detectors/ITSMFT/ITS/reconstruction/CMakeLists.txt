o2_add_library(ITSReconstruction
               SOURCES
               src/ClustererTask.cxx
               src/CookedTracker.cxx
               src/TrivialVertexer.cxx
               PUBLIC_LINK_LIBRARIES
               O2::ITSBase
               O2::ITSMFTReconstruction
               O2::DataFormatsITS)

o2_target_root_dictionary(ITSReconstruction HEADERS
                          include/ITSReconstruction/ClustererTask.h
                          include/ITSReconstruction/CookedTracker.h)
