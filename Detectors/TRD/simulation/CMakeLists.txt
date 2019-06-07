o2_add_library(TRDSimulation SOURCES src/Detector.cxx src/TRsim.cxx
               src/Digitizer.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DetectorsBase
               O2::SimulationDataFormat
               O2::TRDBase)

o2_target_root_dictionary(TRDSimulation HEADERS include/TRDSimulation/Detector.h
                          include/TRDSimulation/TRsim.h
                          include/TRDSimulation/Digitizer.h)
