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

o2_add_library(DataFormatsFV0
  SOURCES src/ChannelData.cxx
          src/Digit.cxx
          src/Hit.cxx
          src/RecPoints.cxx
          src/RawEventData.cxx
          src/CTF.cxx
  PUBLIC_LINK_LIBRARIES O2::FV0Base
                        O2::DataFormatsFIT
                        O2::SimulationDataFormat
                        O2::CommonDataFormat
                        Microsoft.GSL::GSL
)

o2_target_root_dictionary(DataFormatsFV0
  HEADERS include/DataFormatsFV0/ChannelData.h
          include/DataFormatsFV0/Digit.h
          include/DataFormatsFV0/FV0CalibrationInfoObject.h
          include/DataFormatsFV0/FV0ChannelTimeCalibrationObject.h
          include/DataFormatsFV0/MCLabel.h
          include/DataFormatsFV0/Hit.h
          include/DataFormatsFV0/RecPoints.h
          include/DataFormatsFV0/RawEventData.h
          include/DataFormatsFV0/LookUpTable.h
          include/DataFormatsFV0/CTF.h
)
