# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(G3Setup
               SOURCES src/G3Config.cxx
               PUBLIC_LINK_LIBRARIES MC::Geant3 FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(G4Setup
               SOURCES src/G4Config.cxx
               PUBLIC_LINK_LIBRARIES MC::Geant4VMC MC::Geant4 FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(FLUKASetup
               SOURCES src/FlukaConfig.cxx
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(MCReplaySetup
               SOURCES src/MCReplayConfig.cxx
               PUBLIC_LINK_LIBRARIES MC::MCReplay FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(O2TrivialMCEngineSetup
               SOURCES src/O2TrivialMCConfig.cxx
               PUBLIC_LINK_LIBRARIES O2::O2TrivialMC FairRoot::Base O2::SimulationDataFormat O2::Generators O2::SimSetup
)

o2_add_library(SimSetup
               SOURCES  src/SimSetup.cxx src/SetCuts.cxx src/FlukaParam.cxx src/MCReplayParam.cxx
               PUBLIC_LINK_LIBRARIES O2::CommonUtils O2::DetectorsBase O2::SimConfig
)

o2_target_root_dictionary(SimSetup
                          HEADERS include/SimSetup/SimSetup.h
                                  include/SimSetup/FlukaParam.h
                                  include/SimSetup/MCReplayParam.h
                          LINKDEF src/GConfLinkDef.h)

o2_add_test_root_macro(UserDecay.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

o2_add_test_root_macro(commonConfig.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

o2_add_test_root_macro(g3libs.C LABELS simsetup)

o2_add_test_root_macro(g3Config.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

#o2_add_test_root_macro(g4Config.C
#                       PUBLIC_LINK_LIBRARIES O2::SimSetup
#                       LABELS simsetup)


o2_data_file(COPY data  DESTINATION Detectors/gconfig/)
install(FILES src/StandardSteppingTrackRefHook.macro src/FlukaRuntimeConfig.macro DESTINATION share/Detectors/gconfig/)
