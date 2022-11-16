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

if (ENABLE_UPGRADES)
o2_add_executable(digitizer-workflow
                  COMPONENT_NAME sim
                  SOURCES src/CTPDigitizerSpec.cxx
                          src/FT0DigitizerSpec.cxx
                          src/FV0DigitizerSpec.cxx
                          src/FDDDigitizerSpec.cxx
                          src/GRPUpdaterSpec.cxx
                          src/HMPIDDigitizerSpec.cxx
                          src/ITSMFTDigitizerSpec.cxx
                          # src/ITS3DigitizerSpec.cxx
                          src/MCHDigitizerSpec.cxx
                          src/MIDDigitizerSpec.cxx
                          src/PHOSDigitizerSpec.cxx
                          src/CPVDigitizerSpec.cxx
                          src/SimReaderSpec.cxx
                          src/SimpleDigitizerWorkflow.cxx
                          src/TPCDigitRootWriterSpec.cxx
                          src/TPCDigitizerSpec.cxx
                          src/ZDCDigitizerSpec.cxx
                          src/TOFDigitizerSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::Steer
                                        O2::CommonConstants
                                        O2::EMCALSimulation
                                        O2::EMCALWorkflow
                                        O2::FT0Simulation
                                        O2::FV0Simulation
                                        O2::FDDSimulation
                                        O2::CTPSimulation
                                        O2::CTPWorkflowIO
                                        O2::FDDWorkflow
                                        O2::HMPIDSimulation
                                        O2::ITSMFTSimulation
                                        O2::ITSSimulation
                                        O2::ITSMFTWorkflow
                                        O2::MCHSimulation
                                        O2::MCHMappingImpl4
                                        O2::DataFormatsMCH
                                        O2::MFTSimulation
                                        O2::MIDSimulation
                                        O2::PHOSSimulation
                                        O2::CPVSimulation
                                        O2::TOFSimulation
                                        O2::TOFCalibration
                                        O2::TOFReconstruction
                                        O2::TOFWorkflowIO
                                        O2::TPCSimulation
                                        O2::TRDSimulation
                                        O2::TRDWorkflow
                                        O2::TRDWorkflowIO
                                        O2::DataFormatsTRD
                                        O2::ZDCSimulation
                                        O2::ZDCWorkflow
                                        O2::DetectorsRaw
                                        # O2::ITS3Simulation
                                        # O2::ITS3Workflow
                                        )
else()
o2_add_executable(digitizer-workflow
                  COMPONENT_NAME sim
                  SOURCES src/CTPDigitizerSpec.cxx
                          src/FT0DigitizerSpec.cxx
                          src/FV0DigitizerSpec.cxx
                          src/FDDDigitizerSpec.cxx
                          src/GRPUpdaterSpec.cxx
                          src/HMPIDDigitizerSpec.cxx
                          src/ITSMFTDigitizerSpec.cxx
                          src/MCHDigitizerSpec.cxx
                          src/MIDDigitizerSpec.cxx
                          src/PHOSDigitizerSpec.cxx
                          src/CPVDigitizerSpec.cxx
                          src/SimReaderSpec.cxx
                          src/SimpleDigitizerWorkflow.cxx
                          src/TPCDigitRootWriterSpec.cxx
                          src/TPCDigitizerSpec.cxx
                          src/ZDCDigitizerSpec.cxx
                          src/TOFDigitizerSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::Steer
                                        O2::CommonConstants
                                        O2::EMCALSimulation
                                        O2::EMCALWorkflow
                                        O2::FT0Simulation
                                        O2::FV0Simulation
                                        O2::FDDSimulation
                                        O2::CTPSimulation
                                        O2::CTPWorkflowIO
                                        O2::FDDWorkflow
                                        O2::HMPIDSimulation
                                        O2::ITSMFTSimulation
                                        O2::ITSSimulation
                                        O2::ITSMFTWorkflow
                                        O2::MCHSimulation
                                        O2::MCHMappingImpl4
                                        O2::MFTSimulation
                                        O2::MIDSimulation
                                        O2::PHOSSimulation
                                        O2::CPVSimulation
                                        O2::TOFSimulation
                                        O2::TOFCalibration
                                        O2::TOFReconstruction
                                        O2::TOFWorkflowIO
                                        O2::TPCSimulation
                                        O2::TRDSimulation
                                        O2::TRDWorkflow
                                        O2::DataFormatsTRD
                                        O2::ZDCSimulation
                                        O2::ZDCWorkflow
                                        O2::DetectorsRaw
                                        )
endif()


o2_add_executable(mctruth-testworkflow
                  COMPONENT_NAME sim
                  SOURCES src/MCTruthTestWorkflow.cxx
                          src/MCTruthSourceSpec.cxx
                          src/MCTruthWriterSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::SimulationDataFormat)
