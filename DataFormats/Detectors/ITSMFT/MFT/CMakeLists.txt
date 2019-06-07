o2_add_library(DataFormatsMFT
               SOURCES
               src/TrackMFT.cxx
               PUBLIC_LINK_LIBRARIES
               O2::ReconstructionDataFormats
               O2::DataFormatsITSMFT)

o2_target_root_dictionary(DataFormatsMFT HEADERS
                          include/DataFormatsMFT/TrackMFT.h)
