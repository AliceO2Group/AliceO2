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

o2_add_library(FT0Calibration
        SOURCES
        src/CalibParam.cxx
        src/FT0ChannelTimeOffsetSlotContainer.cxx
        src/FT0TimeOffsetSlotContainer.cxx
        src/GlobalOffsetsContainer.cxx
        src/FT0CalibTimeSlewing.cxx
        src/FT0CalibCollector.cxx
        PUBLIC_LINK_LIBRARIES
         O2::CCDB
         O2::MathUtils
         O2::DataFormatsFT0
         O2::CommonDataFormat
         O2::DetectorsCalibration
         O2::DataFormatsGlobalTracking
         ROOT::Minuit
         Microsoft.GSL::GSL
           )
       o2_target_root_dictionary(FT0Calibration
        HEADERS
        include/FT0Calibration/CalibParam.h
        include/FT0Calibration/FT0ChannelTimeOffsetSlotContainer.h
        include/FT0Calibration/FT0TimeOffsetSlotContainer.h
        include/FT0Calibration/GlobalOffsetsContainer.h
        include/FT0Calibration/FT0CalibTimeSlewing.h
        include/FT0Calibration/FT0CalibCollector.h
        )
      o2_add_executable(ft0-channel-offset-calibration
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0ChannelTimeCalibration-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration O2::FITCalibration
        )
      o2_add_executable(ft0-time-offset-calib
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0TimeOffsetCalibration-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration O2::FITCalibration
        )
      o2_add_executable(ft0-tf-processor
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0TFProcessor-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )
      o2_add_executable(ft0-time-spectra-processor
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0TimeSpectraProcessor-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )
      o2_add_executable(ft0-slew-upload
        COMPONENT_NAME calibration
        SOURCES testWorkflow/slew_upload.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )
      o2_add_executable(ft0-global-offsets
        COMPONENT_NAME calibration
        SOURCES testWorkflow/GlobalOffsetsCalibrationWorkflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration O2::FITCalibration
        )
     o2_add_executable(ft0-collect-global-offsets-workflow
        COMPONENT_NAME calibration
        SOURCES testWorkflow/GlobalOffsetsCollectWorkflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )
      o2_add_executable(ft0-collect-calib-workflow
        COMPONENT_NAME calibration
        SOURCES testWorkflow/ft0-collect-calib-workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )

o2_data_file(COPY files DESTINATION Detectors/FT0/)
