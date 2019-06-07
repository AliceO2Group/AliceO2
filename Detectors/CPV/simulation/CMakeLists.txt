o2_add_library(CPVSimulation
               SOURCES
               src/Detector.cxx
               src/GeometryParams.cxx
               src/Digitizer.cxx
               src/DigitizerTask.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DetectorsBase
               O2::CPVBase)

o2_target_root_dictionary(CPVSimulation
                          HEADERS
                          include/CPVSimulation/Detector.h
                          include/CPVSimulation/GeometryParams.h
                          include/CPVSimulation/Digitizer.h
                          include/CPVSimulation/DigitizerTask.h)
