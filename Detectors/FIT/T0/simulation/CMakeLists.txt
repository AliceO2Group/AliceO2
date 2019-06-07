o2_add_library(T0Simulation
               SOURCES
               src/Detector.cxx
               PUBLIC_LINK_LIBRARIES
               O2::SimulationDataFormat
               O2::T0Base
               O2::DataFormatsFITT0)

o2_target_root_dictionary(T0Simulation HEADERS include/T0Simulation/Detector.h)
