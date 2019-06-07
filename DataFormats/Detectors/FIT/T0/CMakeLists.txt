o2_add_library(DataFormatsFITT0
               SOURCES
               src/Digit.cxx
               src/RecPoints.cxx
               PUBLIC_LINK_LIBRARIES
               O2::CommonDataFormat
               O2::SimulationDataFormat)

o2_target_root_dictionary(DataFormatsFITT0
                          HEADERS
                          include/DataFormatsFITT0/Digit.h
                          include/DataFormatsFITT0/RecPoints.h
                          include/DataFormatsFITT0/MCLabel.h
                          include/DataFormatsFITT0/HitType.h)
