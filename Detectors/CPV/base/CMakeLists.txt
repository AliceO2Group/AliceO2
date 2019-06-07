o2_add_library(CPVBase SOURCES src/Geometry.cxx src/Hit.cxx src/Digit.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_target_root_dictionary(CPVBase HEADERS include/CPVBase/Geometry.h
                          include/CPVBase/Hit.h include/CPVBase/Digit.h)
