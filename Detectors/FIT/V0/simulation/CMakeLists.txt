o2_add_library(V0Simulation
               SOURCES
               src/Detector.cxx
               PUBLIC_LINK_LIBRARIES
               O2::SimulationDataFormat
               O2::V0Base
               O2::DataFormatsFITV0
               ROOT::Physics)

o2_target_root_dictionary(V0Simulation HEADERS include/V0Simulation/Detector.h)
