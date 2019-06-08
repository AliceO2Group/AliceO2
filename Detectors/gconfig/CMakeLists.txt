o2_add_library(SimSetup
               SOURCES src/G3Config.cxx src/G4Config.cxx
                       src/GlobalProcessCutSimParam.cxx src/SimSetup.cxx
               PUBLIC_LINK_LIBRARIES geant321
                                     geant4vmc
                                     geant4
                                     O2::SimulationDataFormat
                                     O2::DetectorsPassive
                                     pythia6 # this is needed by Geant3 and
                                             # EGPythia6
                                     ROOT::EGPythia6 # this is needed by Geant4
                                                    # (TPythia6Decayer)
               )

# set(MODULE_NAME "SimSetup") o2_add_test_root_macro(FILENAME "DecayConfig.C")
# o2_add_test_root_macro(FILENAME "UserDecay.C") o2_add_test_root_macro(FILENAME
# "commonConfig.C" DEPENDS_ON SimulationDataFormat)
# o2_add_test_root_macro(FILENAME "g3libs.C") #o2_add_test_root_macro(FILENAME
# "g3Config.C" DEPENDS_ON SimulationDataFormat MC::Geant3)
# #o2_add_test_root_macro(FILENAME "g4Config.C" DEPENDS_ON SimulationDataFormat
# MC::Geant4VMC)
