o2_add_library(TOFSimulation
               SOURCES
               src/Detector.cxx
               src/Digitizer.cxx
               src/DigitizerTask.cxx
               src/Strip.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DetectorsBase
               O2::TOFBase
               O2::SimulationDataFormat)

o2_target_root_dictionary(TOFSimulation
                          HEADERS
                          include/TOFSimulation/Detector.h
                          include/TOFSimulation/Digitizer.h
                          include/TOFSimulation/DigitizerTask.h
                          include/TOFSimulation/Strip.h)
