# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(TRKFT3Simulation
               SOURCES src/ChipDigitsContainer.cxx
                       src/ChipSimResponse.cxx
                       src/DigiParams.cxx
                       src/Digitizer.cxx
                       src/DPLDigitizerParam.cxx
               PUBLIC_LINK_LIBRARIES O2::TRKBase
                                     O2::FT3Base
                                     O2::DataFormatsTRKFT3
                                     O2::ITSMFTSimulation
                                     O2::DetectorsRaw
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(TRKFT3Simulation
                          HEADERS include/TRKFT3Simulation/ChipDigitsContainer.h
                                  include/TRKFT3Simulation/ChipSimResponse.h
                                  include/TRKFT3Simulation/DigiParams.h
                                  include/TRKFT3Simulation/Digitizer.h
                                  include/TRKFT3Simulation/DPLDigitizerParam.h
                          LINKDEF src/TRKFT3SimulationLinkDef.h)
