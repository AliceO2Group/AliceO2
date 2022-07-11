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

o2_add_test_root_macro(
  makeChannelTimeOffsetFV0CalibObjectInCCDB.C
  PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                        O2::FV0Simulation
                        O2::DataFormatsFV0
                        O2::Framework
                        O2::CCDB
                        O2::FV0Calibration
                        O2::DetectorsCalibration
)

o2_add_test_root_macro(
  readChannelTimeOffsetFV0CalibObjectFromCCDB.C
  PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                        O2::FV0Simulation
                        O2::DataFormatsFV0
                        O2::Framework
                        O2::CCDB
                        O2::FV0Calibration
                        O2::DetectorsCalibration
)

install(
  FILES makeChannelTimeOffsetFV0CalibObjectInCCDB.C
        readChannelTimeOffsetFV0CalibObjectFromCCDB.C
  DESTINATION share/macro/
)
