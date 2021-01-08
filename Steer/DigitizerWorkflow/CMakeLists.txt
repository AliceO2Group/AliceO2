# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.


o2_add_executable(digitizer-workflow
                  COMPONENT_NAME sim
                  SOURCES src/EMCALDigitWriterSpec.cxx
                          src/EMCALDigitizerSpec.cxx
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
                                        O2::FT0Simulation
                                        O2::FV0Simulation
                                        O2::FDDSimulation
                                        O2::HMPIDSimulation
                                        O2::ITSMFTSimulation
                                        O2::ITSSimulation
                                        O2::ITSMFTWorkflow
                                        O2::MCHSimulation
                                        O2::MCHBase
                                        O2::MFTSimulation
                                        O2::MIDSimulation
                                        O2::PHOSSimulation
                                        O2::CPVSimulation
                                        O2::TOFSimulation
                                        O2::TOFCalibration
                                        O2::TOFReconstruction
                                        O2::TOFWorkflowUtils
                                        O2::TPCSimulation
                                        O2::TRDSimulation
                                        O2::TRDWorkflow
                                        O2::DataFormatsTRD
                                        O2::ZDCSimulation)


o2_add_executable(mctruth-testworkflow
                  COMPONENT_NAME sim
                  SOURCES src/MCTruthTestWorkflow.cxx
		          src/MCTruthSourceSpec.cxx
	                  src/MCTruthWriterSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::SimulationDataFormat)
