# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITSMFTSimulation
               SOURCES src/Hit.cxx
                       src/AlpideSimResponse.cxx
                       src/ChipDigitsContainer.cxx
                       src/AlpideChip.cxx
                       src/DigiParams.cxx
                       src/Digitizer.cxx
                       src/AlpideSignalTrapezoid.cxx
                       src/ClusterShape.cxx
                       src/DPLDigitizerParam.cxx
               PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::ITSMFTBase
                                     O2::DataFormatsITSMFT)

o2_target_root_dictionary(
  ITSMFTSimulation
  HEADERS include/ITSMFTSimulation/Hit.h
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
            SOURCES test/testAlpideSimResponse.cxx
            COMPONENT_NAME ITSMFT
            PUBLIC_LINK_LIBRARIES O2::ITSMFTSimulation
            LABELS "its;mft"
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
