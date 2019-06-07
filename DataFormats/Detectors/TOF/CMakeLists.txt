o2_add_library(DataFormatsTOF
               SOURCES
               src/Cluster.cxx
               PUBLIC_LINK_LIBRARIES
               O2::ReconstructionDataFormats
               Boost::serialization)

o2_target_root_dictionary(DataFormatsTOF HEADERS
                          include/DataFormatsTOF/Cluster.h)
