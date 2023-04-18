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

o2_add_library(MCHBase
         SOURCES
           src/Error.cxx
           src/ErrorMap.cxx
           src/MathiesonOriginal.cxx
           src/PreCluster.cxx
           src/SanityCheck.cxx
           src/ResponseParam.cxx
           src/TrackBlock.cxx
           src/Trackable.cxx
           src/TrackerParam.cxx
         PUBLIC_LINK_LIBRARIES O2::DataFormatsMCH)

o2_target_root_dictionary(MCHBase
                          HEADERS include/MCHBase/DecoderError.h
                                  include/MCHBase/Error.h
                                  include/MCHBase/HeartBeatPacket.h
                                  include/MCHBase/PreCluster.h
                                  include/MCHBase/ResponseParam.h
                                  include/MCHBase/TrackerParam.h)

o2_add_test(trackable
            SOURCES src/testTrackable.cxx
            COMPONENT_NAME mch
            PUBLIC_LINK_LIBRARIES O2::MCHBase
            LABELS muon;mch)

o2_add_test(error
            SOURCES src/testError.cxx
            COMPONENT_NAME mch
            PUBLIC_LINK_LIBRARIES O2::MCHBase
            LABELS muon;mch)

o2_add_test(errormap
            SOURCES src/testErrorMap.cxx
            COMPONENT_NAME mch
            PUBLIC_LINK_LIBRARIES O2::MCHBase
            LABELS muon;mch)
