o2_add_library(HMPIDSimulation
               SOURCES
               src/Detector.cxx
               src/HMPIDDigitizer.cxx
               PUBLIC_LINK_LIBRARIES
               O2::HMPIDBase)

o2_target_root_dictionary(HMPIDSimulation HEADERS
                          include/HMPIDSimulation/Detector.h
                          include/HMPIDSimulation/HMPIDDigitizer.h)
