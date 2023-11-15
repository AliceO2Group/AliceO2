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

o2_add_library(DataFormatsMCH
               SOURCES src/TrackMCH.cxx
                       src/Digit.cxx
                       src/CTF.cxx
                       src/ROFRecord.cxx
                       src/Cluster.cxx
                       src/DsChannelId.cxx
            PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
                                     O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(DataFormatsMCH
                          HEADERS include/DataFormatsMCH/Digit.h
                                  include/DataFormatsMCH/DsChannelGroup.h
                                  include/DataFormatsMCH/ROFRecord.h
                                  include/DataFormatsMCH/TrackMCH.h
                                  include/DataFormatsMCH/Cluster.h
                                  include/DataFormatsMCH/CTF.h)

o2_add_executable(convert-bad-channels
                  SOURCES src/convert-bad-channels.cxx
                  COMPONENT_NAME mch
                  PUBLIC_LINK_LIBRARIES O2::DataFormatsMCH Boost::program_options)

o2_add_test(digit
            SOURCES src/testDigit.cxx
            PUBLIC_LINK_LIBRARIES O2::MCHBase
            COMPONENT_NAME mch
            LABELS muon;mch)

o2_add_test(track
            SOURCES src/testTrackMCH.cxx
            PUBLIC_LINK_LIBRARIES O2::MCHBase
            COMPONENT_NAME mch
            LABELS muon;mch)

