o2_add_library(MFTReconstruction
               SOURCES
               src/ClustererTask.cxx
               src/TrackerTask.cxx
               src/Tracker.cxx
               PUBLIC_LINK_LIBRARIES
               O2::MFTBase
               O2::ITSMFTReconstruction
               O2::DataFormatsMFT)

o2_target_root_dictionary(MFTReconstruction HEADERS
                          include/MFTReconstruction/ClustererTask.h
                          include/MFTReconstruction/TrackerTask.h)
