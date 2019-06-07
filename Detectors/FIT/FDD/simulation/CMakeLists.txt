o2_add_library(FDDSimulation SOURCES src/Detector.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
        O2::FDDBase ROOT::Physics)

o2_target_root_dictionary(FDDSimulation HEADERS include/FDDSimulation/Detector.h
                          include/FDDSimulation/Hit.h)
