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

o2_add_library(TRKSimulation
               SOURCES src/Hit.cxx
                       src/TRKLayer.cxx
                       src/ChipDigitsContainer.cxx
                       src/ChipSimResponse.cxx
                       src/Detector.cxx
                       src/DigiParams.cxx
                       src/Digitizer.cxx
                       src/TRKServices.cxx
                       src/DPLDigitizerParam.cxx
                       src/VDLayer.cxx
                       src/VDGeometryBuilder.cxx
               PUBLIC_LINK_LIBRARIES O2::TRKBase
                                     O2::FT3Simulation
                                     O2::ITSMFTSimulation
                                     O2::DetectorsRaw
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(TRKSimulation
                          HEADERS include/TRKSimulation/Hit.h
                                  include/TRKSimulation/ChipDigitsContainer.h
                                  include/TRKSimulation/ChipSimResponse.h
                                  include/TRKSimulation/DigiParams.h
                                  include/TRKSimulation/Digitizer.h
                                  include/TRKSimulation/Detector.h
                                  include/TRKSimulation/TRKLayer.h
                                  include/TRKSimulation/TRKServices.h
                                  include/TRKSimulation/VDLayer.h
                                  include/TRKSimulation/VDGeometryBuilder.h
                                  include/TRKSimulation/VDSensorRegistry.h
                                  include/TRKSimulation/DPLDigitizerParam.h)
