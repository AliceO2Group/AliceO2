o2_add_library(DataFormatsFITV0 SOURCES src/Hit.cxx PUBLIC_LINK_LIBRARIES
               O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsFITV0 HEADERS
                          include/DataFormatsFITV0/Hit.h)
