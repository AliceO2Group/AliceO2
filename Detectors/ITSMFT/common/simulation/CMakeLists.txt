o2_add_library(ITSMFTSimulation
               SOURCES
               src/Hit.cxx
               src/AlpideSimResponse.cxx
               src/ChipDigitsContainer.cxx
               src/AlpideChip.cxx
               src/DigiParams.cxx
               src/Digitizer.cxx
               src/AlpideSignalTrapezoid.cxx
               src/ClusterShape.cxx
               src/DPLDigitizerParam.cxx
               PUBLIC_LINK_LIBRARIES
               O2::SimulationDataFormat
               O2::ITSMFTBase
               O2::DataFormatsITSMFT)

o2_target_root_dictionary(ITSMFTSimulation
                          HEADERS
                          include/ITSMFTSimulation/Hit.h
                          include/ITSMFTSimulation/AlpideSimResponse.h
                          include/ITSMFTSimulation/DigiParams.h
                          include/ITSMFTSimulation/Digitizer.h
                          include/ITSMFTSimulation/PreDigit.h
                          include/ITSMFTSimulation/ChipDigitsContainer.h
                          include/ITSMFTSimulation/AlpideChip.h
                          include/ITSMFTSimulation/AlpideSignalTrapezoid.h
                          include/ITSMFTSimulation/ClusterShape.h
                          include/ITSMFTSimulation/DPLDigitizerParam.h)

o2_add_test(AlpideSimResponse
            SOURCES
            test/testAlpideSimResponse.cxx
            COMPONENT_NAME
            ITSMFT
            PUBLIC_LINK_LIBRARIES
            O2::ITSMFTSimulation
            LABELS "its;mft")
