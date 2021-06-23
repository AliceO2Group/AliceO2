# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DataFormatsFT0
  SOURCES src/Digit.cxx
  SOURCES src/DigitsTemp.cxx
  SOURCES src/ChannelData.cxx
  SOURCES src/RecPoints.cxx
  SOURCES src/RawEventData.cxx
          src/CTF.cxx
  PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                        O2::Headers
                        O2::SimulationDataFormat
                        O2::CCDB
                        O2::DetectorsCalibration
                        O2::FT0Base
                        O2::DataFormatsFIT)

o2_target_root_dictionary(DataFormatsFT0
                          HEADERS include/DataFormatsFT0/Digit.h
                                  include/DataFormatsFT0/DigitsTemp.h
                                  include/DataFormatsFT0/ChannelData.h
                                  include/DataFormatsFT0/RecPoints.h
                                  include/DataFormatsFT0/MCLabel.h
                                  include/DataFormatsFT0/HitType.h
                                  include/DataFormatsFT0/RawEventData.h
                                  include/DataFormatsFT0/LookUpTable.h
                                  include/DataFormatsFT0/CTF.h)
