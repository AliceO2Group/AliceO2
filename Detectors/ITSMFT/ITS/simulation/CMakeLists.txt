o2_add_library(ITSSimulation
               SOURCES src/V11Geometry.cxx src/V1Layer.cxx src/V3Layer.cxx
                       src/Detector.cxx src/V3Services.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSBase O2::ITSMFTSimulation
                                     ROOT::Physics)

o2_target_root_dictionary(ITSSimulation
                          HEADERS include/ITSSimulation/Detector.h
                                  include/ITSSimulation/V1Layer.h
                                  include/ITSSimulation/V3Layer.h
                                  include/ITSSimulation/V11Geometry.h
                                  include/ITSSimulation/V3Services.h)
