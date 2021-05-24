# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(PHOSCalibWorkflow
               SOURCES src/PHOSPedestalCalibDevice.cxx
                       src/PHOSHGLGRatioCalibDevice.cxx
                       src/PHOSEnergyCalibrator.cxx
                       src/PHOSEnergyCalibDevice.cxx
                       src/TurnOnHistos.cxx
                       src/PHOSTurnonCalibDevice.cxx
                       src/PHOSTurnonCalibrator.cxx
                       src/PHOSRunbyrunCalibrator.cxx
                       src/PHOSRunbyrunCalibDevice.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsPHOS
                                     O2::DetectorsRaw
                                     O2::PHOSReconstruction
                                     O2::DetectorsCalibration)
o2_add_header_only_library(PHOSCalib)

o2_target_root_dictionary(PHOSCalibWorkflow
                          HEADERS include/PHOSCalibWorkflow/PHOSPedestalCalibDevice.h
                                  include/PHOSCalibWorkflow/PHOSHGLGRatioCalibDevice.h
                                  include/PHOSCalibWorkflow/PHOSTurnonCalibrator.h
                                  include/PHOSCalibWorkflow/RingBuffer.h
                                  include/PHOSCalibWorkflow/ETCalibHistos.h
                                  include/PHOSCalibWorkflow/PHOSEnergyCalibrator.h
                                  include/PHOSCalibWorkflow/PHOSEnergyCalibDevice.h
                                  include/PHOSCalibWorkflow/TurnOnHistos.h
                                  include/PHOSCalibWorkflow/PHOSTurnonCalibrator.h
                                  include/PHOSCalibWorkflow/PHOSTurnonCalibDevice.h
                                  include/PHOSCalibWorkflow/PHOSRunbyrunCalibrator.h
                                  include/PHOSCalibWorkflow/PHOSRunbyrunCalibDevice.h
                          LINKDEF src/PHOSCalibWorkflowLinkDef.h)

o2_add_executable(calib-workflow
                  COMPONENT_NAME phos
                  SOURCES src/phos-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::DataFormatsPHOS
                                        O2::PHOSCalibWorkflow
                                        O2::DetectorsCalibration)


