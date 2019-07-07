# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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

o2_target_root_dictionary(SimSetup
                          HEADERS include/SimSetup/SimSetup.h
                                  include/SimSetup/GlobalProcessCutSimParam.h
                          LINKDEF src/GConfLinkDef.h)

o2_add_test_root_macro(DecayConfig.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)

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

o2_add_test_root_macro(g4Config.C
                       PUBLIC_LINK_LIBRARIES O2::SimSetup
                       LABELS simsetup)
