o2_add_library(PHOSBase SOURCES src/Geometry.cxx src/Hit.cxx src/Digit.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat)

o2_target_root_dictionary(PHOSBase HEADERS include/PHOSBase/Geometry.h
                          include/PHOSBase/Hit.h include/PHOSBase/Digit.h)
