add_library(SimSetup SHARED src/G3Config.cxx src/G4Config.cxx
                            src/GlobalProcessCutSimParam.cxx src/SimSetup.cxx)

target_link_libraries(SimSetup
                      PUBLIC Geant3
                             Geant4VMC
                             SimulationDataFormat
                             DetectorsPassive
                             pythia6 # this is needed by Geant3 and EGPythia6
                             EGPythia6 # this is needed by Geant4
                                       # (TPythia6Decayer)
                      )

# set(MODULE_NAME "SimSetup")
#
# O2_SETUP(NAME ${MODULE_NAME})
#
# set(SRCS src/G3Config.cxx src/G4Config.cxx src/SimSetup.cxx
# src/GlobalProcessCutSimParam.cxx )
#
# set(HEADERS include/${MODULE_NAME}/SimSetup.h
# include/${MODULE_NAME}/GlobalProcessCutSimParam.h )
#
# Set(LINKDEF src/GConfLinkDef.h) set(LIBRARY_NAME ${MODULE_NAME})
# set(BUCKET_NAME simulation_setup_bucket)
#
# O2_GENERATE_LIBRARY()
#
# o2_add_test_root_macro(FILENAME "DecayConfig.C")
# o2_add_test_root_macro(FILENAME "UserDecay.C") o2_add_test_root_macro(FILENAME
# "commonConfig.C" DEPENDS_ON SimulationDataFormat)
# o2_add_test_root_macro(FILENAME "g3libs.C") #o2_add_test_root_macro(FILENAME
# "g3Config.C" DEPENDS_ON SimulationDataFormat MC::Geant3)
# #o2_add_test_root_macro(FILENAME "g4Config.C" DEPENDS_ON SimulationDataFormat
# MC::Geant4VMC)
