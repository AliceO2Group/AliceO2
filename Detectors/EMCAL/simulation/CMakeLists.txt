o2_add_library(EMCALSimulation
               SOURCES
               src/Detector.cxx
               src/Digitizer.cxx
               src/DigitizerTask.cxx
               src/SpaceFrame.cxx
               src/SimParam.cxx
               PUBLIC_LINK_LIBRARIES
               O2::EMCALBase
               O2::DetectorsBase)

o2_target_root_dictionary(EMCALSimulation
                          HEADERS
                          include/EMCALSimulation/Detector.h
                          include/EMCALSimulation/Digitizer.h
                          include/EMCALSimulation/DigitizerTask.h
                          include/EMCALSimulation/SpaceFrame.h
                          include/EMCALSimulation/SimParam.h)
