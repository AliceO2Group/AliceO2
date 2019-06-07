o2_add_library(FITSimulation SOURCES src/Digitizer.cxx
        PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
        O2::DataFormatsFITT0
        O2::T0Simulation)

o2_target_root_dictionary(FITSimulation HEADERS
                          include/FITSimulation/Digitizer.h
                          include/FITSimulation/DigitizationParameters.h)
