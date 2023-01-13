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

o2_add_library(MCHTracking
        SOURCES
           src/TrackParam.cxx
           src/Track.cxx
           src/TrackExtrap.cxx
           src/TrackFitter.cxx
           src/TrackFinderOriginal.cxx
           src/TrackFinder.cxx
           src/TrackFinderSpec.cxx
        PUBLIC_LINK_LIBRARIES
           O2::Field
           O2::DetectorsBase
           O2::MCHBase
           O2::Framework
           O2::CommonUtils
           O2::DataFormatsParameters)

o2_add_executable(
        clusters-to-tracks-workflow
        SOURCES src/clusters-to-tracks-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHTracking)

o2_add_executable(
        tracks-to-tracks-workflow
        SOURCES src/TrackFitterSpec.cxx src/tracks-to-tracks-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHTracking)
