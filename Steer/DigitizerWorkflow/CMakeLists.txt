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
                  SOURCES src/CollisionTimePrinter.cxx
                          src/EMCALDigitWriterSpec.cxx
                          src/EMCALDigitizerSpec.cxx
                          src/FITDigitWriterSpec.cxx
                          src/FITDigitizerSpec.cxx
			  src/FDDDigitizerSpec.cxx
                          src/GRPUpdaterSpec.cxx
                          src/HMPIDDigitizerSpec.cxx
                          src/ITSMFTDigitWriterSpec.cxx
                          src/ITSMFTDigitizerSpec.cxx
                          src/MCHDigitizerSpec.cxx
                          src/MIDDigitizerSpec.cxx
                          src/PHOSDigitizerSpec.cxx
                          src/PHOSDigitWriterSpec.cxx
                          src/SimReaderSpec.cxx
                          src/SimpleDigitizerWorkflow.cxx
                          src/TOFDigitWriterSpec.cxx
                          src/TOFDigitizerSpec.cxx
                          src/TPCDigitRootWriterSpec.cxx
                          src/TPCDigitizerSpec.cxx
                          src/TRDDigitizerSpec.cxx
                          src/ZDCDigitizerSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::Steer
                                        O2::EMCALSimulation
                                        O2::FITSimulation
					O2::FDDSimulation
                                        O2::HMPIDSimulation
                                        O2::ITSMFTSimulation
                                        O2::ITSSimulation
                                        O2::MCHSimulation
                                        O2::MFTSimulation
                                        O2::MIDSimulation
                                        O2::PHOSSimulation
                                        O2::TOFSimulation
                                        O2::TOFReconstruction
                                        O2::TPCSimulation
                                        O2::TPCWorkflow
                                        O2::TRDSimulation
                                        O2::ZDCSimulation)
