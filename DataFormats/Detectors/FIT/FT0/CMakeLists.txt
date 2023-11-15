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

o2_add_library(DataFormatsFT0
  SOURCES src/ChannelData.cxx
          src/Digit.cxx
          src/DigitFilterParam.cxx
          src/CalibParam.cxx
          src/RecPoints.cxx
          src/RawEventData.cxx
          src/CTF.cxx
          src/LookUpTable.cxx
  PUBLIC_LINK_LIBRARIES O2::FT0Base
                        O2::DataFormatsFIT
                        O2::SimulationDataFormat
                        O2::CommonDataFormat
                        O2::Headers
                        O2::CCDB
)

o2_target_root_dictionary(DataFormatsFT0
  HEADERS include/DataFormatsFT0/ChannelData.h
          include/DataFormatsFT0/Digit.h
          include/DataFormatsFT0/DigitFilterParam.h
          include/DataFormatsFT0/CalibParam.h
          include/DataFormatsFT0/MCLabel.h
          include/DataFormatsFT0/HitType.h
          include/DataFormatsFT0/RecPoints.h
          include/DataFormatsFT0/RawEventData.h
          include/DataFormatsFT0/LookUpTable.h
          include/DataFormatsFT0/CTF.h
          include/DataFormatsFT0/RecoCalibInfoObject.h
          include/DataFormatsFT0/FT0CalibrationInfoObject.h
          include/DataFormatsFT0/FT0ChannelTimeCalibrationObject.h
          include/DataFormatsFT0/GlobalOffsetsInfoObject.h
          include/DataFormatsFT0/GlobalOffsetsCalibrationObject.h
          include/DataFormatsFT0/SpectraInfoObject.h
)
