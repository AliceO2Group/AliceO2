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
        src/FT0TimeOffsetSlotContainer.cxx
        PUBLIC_LINK_LIBRARIES
         O2::DataFormatsFT0
         O2::CommonDataFormat
         O2::DetectorsCalibration
           )
       o2_target_root_dictionary(FT0Calibration
        HEADERS
        include/FT0Calibration/FT0TimeOffsetSlotContainer.h
        )
      o2_add_executable(ft0-time-offset-calib
        COMPONENT_NAME calibration
        SOURCES workflow/FT0TimeOffsetCalibration-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration O2::FITCalibration
        )
      o2_add_executable(ft0-time-spectra-processor
        COMPONENT_NAME calibration
        SOURCES workflow/FT0TimeSpectraProcessor-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        )
