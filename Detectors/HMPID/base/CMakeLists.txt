o2_add_library(HMPIDBase
               SOURCES
               src/Param.cxx
               src/Digit.cxx
               PUBLIC_LINK_LIBRARIES
               O2::CommonDataFormat
               O2::SimulationDataFormat
               ROOT::Physics)

o2_target_root_dictionary(HMPIDBase HEADERS include/HMPIDBase/Param.h
                          include/HMPIDBase/Digit.h)
