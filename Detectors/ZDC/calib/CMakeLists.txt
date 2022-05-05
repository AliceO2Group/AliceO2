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

o2_add_library(ZDCCalib
               SOURCES
                 src/InterCalib.cxx
                 src/InterCalibConfig.cxx
                 src/InterCalibSpec.cxx
                 src/InterCalibEPN.cxx
                 src/InterCalibEPNSpec.cxx
                 src/WaveformCalibConfig.cxx
               PUBLIC_LINK_LIBRARIES
                 O2::CCDB
                 O2::DPLUtils
                 O2::DataFormatsZDC
                 O2::DetectorsRaw
                 O2::Headers
                 O2::SimConfig
                 O2::SimulationDataFormat
                 O2::ZDCBase
                 O2::ZDCSimulation
                 O2::ZDCReconstruction
                 O2::DetectorsCalibration
                 ROOT::Core
                 ROOT::Hist
                 ROOT::Minuit)

o2_target_root_dictionary(ZDCCalib
                          HEADERS
                            include/ZDCCalib/InterCalib.h
                            include/ZDCCalib/InterCalibConfig.h
                            include/ZDCCalib/InterCalibSpec.h
                            include/ZDCCalib/InterCalibEPN.h
                            include/ZDCCalib/InterCalibEPNSpec.h
                            include/ZDCCalib/WaveformCalibConfig.h)

o2_add_executable(intercalib-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-intercalib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(intercalib-epn-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-intercalib-epn-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)
