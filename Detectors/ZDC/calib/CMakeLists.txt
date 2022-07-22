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
                 src/BaselineCalibData.cxx
                 src/BaselineCalibConfig.cxx
                 src/BaselineCalibEPN.cxx
                 src/BaselineCalibEPNSpec.cxx
                 src/BaselineCalib.cxx
                 src/BaselineCalibSpec.cxx
                 src/InterCalibData.cxx
                 src/InterCalib.cxx
                 src/InterCalibConfig.cxx
                 src/InterCalibSpec.cxx
                 src/InterCalibEPN.cxx
                 src/InterCalibEPNSpec.cxx
                 src/TDCCalibConfig.cxx
                 src/TDCCalibEPN.cxx
                 src/TDCCalibEPNSpec.cxx
                 src/TDCCalibSpec.cxx
                 src/TDCCalibData.cxx
                 src/TDCCalib.cxx
                 src/WaveformCalibConfig.cxx
                 src/WaveformCalib.cxx
                 src/WaveformCalibSpec.cxx
                 src/WaveformCalibEPN.cxx
                 src/WaveformCalibEPNSpec.cxx
                 src/WaveformCalibQueue.cxx
                 src/WaveformCalibData.cxx
                 src/WaveformCalibParam.cxx
                 src/CalibParamZDC.cxx
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
                            include/ZDCCalib/BaselineCalibData.h
                            include/ZDCCalib/BaselineCalibConfig.h
                            include/ZDCCalib/InterCalibData.h
                            include/ZDCCalib/InterCalibConfig.h
                            include/ZDCCalib/TDCCalibConfig.h
                            include/ZDCCalib/TDCCalibData.h
                            include/ZDCCalib/WaveformCalibConfig.h
                            include/ZDCCalib/WaveformCalibData.h
                            include/ZDCCalib/WaveformCalibParam.h
                            include/ZDCCalib/CalibParamZDC.h)

o2_add_executable(intercalib-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-intercalib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(intercalib-epn-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-intercalib-epn-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(waveformcalib-epn-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-waveformcalib-epn-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(waveformcalib-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-waveformcalib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(baselinecalib-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-baselinecalib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(tdccalib-epn-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-tdccalib-epn-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)

o2_add_executable(tdccalib-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-tdccalib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow O2::ZDCCalib O2::DetectorsCalibration)
