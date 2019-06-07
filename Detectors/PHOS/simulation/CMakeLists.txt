o2_add_library(PHOSSimulation
               SOURCES
               src/Detector.cxx
               src/GeometryParams.cxx
               src/Digitizer.cxx
               src/DigitizerTask.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase
               O2::PHOSBase)

o2_target_root_dictionary(PHOSSimulation
                          HEADERS
                          include/PHOSSimulation/Detector.h
                          include/PHOSSimulation/GeometryParams.h
                          include/PHOSSimulation/Digitizer.h
                          include/PHOSSimulation/DigitizerTask.h)
