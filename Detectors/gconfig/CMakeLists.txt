# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(G3Setup
               SOURCES src/G3Config.cxx
               PUBLIC_LINK_LIBRARIES MC::Geant3 FairRoot::Base O2::SimulationDataFormat O2::Generators
)

o2_add_library(G4Setup
               SOURCES src/G4Config.cxx
               PUBLIC_LINK_LIBRARIES MC::Geant4VMC MC::Geant4 FairRoot::Base O2::SimulationDataFormat O2::Generators ROOT::EGPythia6
)

if (FlukaVMC_FOUND)
  message(STATUS "BUILDING WITH FLUKA")
  o2_add_library(FLUKASetup
                 SOURCES src/FlukaConfig.cxx
       	         PUBLIC_LINK_LIBRARIES MC::FlukaVMC FairRoot::Base O2::SimulationDataFormat O2::Generators ROOT::EGPythia6 O2::SimSetup
	        )
else()
  message(STATUS "BUILDING WITHOUT FLUKA")
endif (FlukaVMC_FOUND)
	      
o2_add_library(SimSetup
               SOURCES  src/GlobalProcessCutSimParam.cxx src/SimSetup.cxx src/SetCuts.cxx src/FlukaParam.cxx
	       PUBLIC_LINK_LIBRARIES O2::CommonUtils O2::DetectorsBase
              )

o2_target_root_dictionary(SimSetup
                          HEADERS include/SimSetup/SimSetup.h
                                  include/SimSetup/GlobalProcessCutSimParam.h
				  include/SimSetup/FlukaParam.h
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

o2_data_file(COPY data  DESTINATION Detectors/gconfig/)
