o2_add_library(ZDCSimulation
               SOURCES src/Detector.cxx src/Digitizer.cxx
               PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::ZDCBase)

o2_target_root_dictionary(ZDCSimulation
                          HEADERS include/ZDCSimulation/Hit.h
                                  include/ZDCSimulation/Digit.h
                                  include/ZDCSimulation/Digitizer.h
                                  include/ZDCSimulation/Detector.h)

o2_data_file(COPY data DESTINATION Detectors/ZDC/simulation)
