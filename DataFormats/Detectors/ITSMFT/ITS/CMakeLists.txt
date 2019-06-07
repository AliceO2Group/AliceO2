o2_add_library(DataFormatsITS
               SOURCES
               src/TrackITS.cxx
               PUBLIC_LINK_LIBRARIES
               O2::ReconstructionDataFormats
               O2::DataFormatsITSMFT)

o2_target_root_dictionary(DataFormatsITS HEADERS
                          include/DataFormatsITS/TrackITS.h)
