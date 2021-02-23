o2_add_library(DataFormatsCTP
               SOURCES src/CTPRawData.cxx 
                PUBLIC_LINK_LIBRARIES O2::CommonDataFormat 
                                     O2::Headers 
                                     O2::MathUtils 
                                     O2::DetectorsBase 
                                     O2::SimulationDataFormat
                                     Boost::serialization)
o2_target_root_dictionary(DataFormatsCTP
                          HEADERS include/DataFormatsCPT/CTPRawData.h
)
