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

o2_add_library(FV0Calibration
  SOURCES src/FV0ChannelTimeOffsetSlotContainer.cxx
          src/FV0CalibCollector.cxx
  PUBLIC_LINK_LIBRARIES O2::DataFormatsFV0
                        O2::FV0Base
                        O2::CCDB
                        O2::DetectorsCalibration
                        ROOT::Minuit
                        Microsoft.GSL::GSL
)

o2_target_root_dictionary(FV0Calibration
  HEADERS include/FV0Calibration/FV0ChannelTimeOffsetSlotContainer.h
          include/FV0Calibration/FV0CalibCollector.h
)

o2_add_executable(fv0-channel-offset-calibration
  COMPONENT_NAME calibration
  SOURCES workflow/FV0ChannelTimeCalibration-Workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::FV0Calibration O2::FITCalibration
)

o2_add_executable(fv0-tf-processor
  COMPONENT_NAME calibration
  SOURCES workflow/FV0TFProcessor-Workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::FV0Calibration
)

add_subdirectory(macros)
