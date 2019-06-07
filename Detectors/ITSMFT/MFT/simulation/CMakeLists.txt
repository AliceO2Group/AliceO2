o2_add_library(MFTSimulation
               SOURCES
               src/Detector.cxx
               src/DigitizerTask.cxx
               PUBLIC_LINK_LIBRARIES
               O2::MFTBase
               ROOT::Physics)

o2_target_root_dictionary(MFTSimulation HEADERS include/MFTSimulation/Detector.h
                          include/MFTSimulation/DigitizerTask.h)
