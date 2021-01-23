# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CPVCalibWorkflow
               SOURCES src/CPVPedestalCalibDevice.cxx
                       src/CPVGainCalibDevice.cxx
                       src/CPVBadMapCalibDevice.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework 
                                     O2::DataFormatsCPV
                                     O2::DetectorsRaw
                                     O2::CPVReconstruction
                                     O2::CPVCalib 
                                     O2::DetectorsCalibration)


o2_add_executable(calib-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/cpv-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::DataFormatsCPV
                                        O2::CPVCalib
                                        O2::CPVCalibWorkflow
                                        O2::DetectorsCalibration)
