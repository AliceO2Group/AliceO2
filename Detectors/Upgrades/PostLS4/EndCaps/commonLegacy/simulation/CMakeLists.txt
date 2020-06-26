# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EndCapsSimulation
               SOURCES src/Hit.cxx
                       src/AlpideSimResponse.cxx
                       src/ChipDigitsContainer.cxx
                       src/AlpideChip.cxx
                       src/DigiParams.cxx
                       src/Digitizer.cxx
                       src/AlpideSignalTrapezoid.cxx
                       src/ClusterShape.cxx
                       src/DPLDigitizerParam.cxx
		       src/MC2RawEncoder.cxx
		PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::EndCapsBase
		                      O2::EndCapsReconstruction
                                      O2::DataFormatsITSMFT O2::DetectorsRaw)

o2_target_root_dictionary(
  EndCapsSimulation
  HEADERS include/EndCapsSimulation/Hit.h
          include/EndCapsSimulation/AlpideSimResponse.h
          include/EndCapsSimulation/DigiParams.h
          include/EndCapsSimulation/Digitizer.h
          include/EndCapsSimulation/PreDigit.h
          include/EndCapsSimulation/ChipDigitsContainer.h
          include/EndCapsSimulation/AlpideChip.h
          include/EndCapsSimulation/AlpideSignalTrapezoid.h
          include/EndCapsSimulation/ClusterShape.h
          include/EndCapsSimulation/DPLDigitizerParam.h
	  include/EndCapsSimulation/MC2RawEncoder.h
	  )

o2_add_test(AlpideSimResponse
            SOURCES test/testAlpideSimResponse.cxx
            COMPONENT_NAME ITSMFT
            PUBLIC_LINK_LIBRARIES O2::EndCapsSimulation
            LABELS "its;mft"
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
