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

o2_add_library(GlobalTrackingStudy
               SOURCES src/TPCTrackStudy.cxx
                       src/TrackingStudy.cxx
                       src/TPCDataFilter.cxx
                       src/ITSOffsStudy.cxx
                       src/DumpTracks.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTracking
                                     O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflowHelpers
                                     O2::DataFormatsGlobalTracking
                                     O2::DetectorsVertexing
                                     O2::SimulationDataFormat)

o2_add_executable(study-workflow
                  COMPONENT_NAME tpc-track
                  SOURCES src/tpc-track-study-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingStudy)

o2_add_executable(study-workflow
                  COMPONENT_NAME tracking
                  SOURCES src/tracking-study-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingStudy)

o2_add_executable(filter-workflow
                  COMPONENT_NAME tpc-data
                  SOURCES src/tpc-data-filter-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingStudy)

o2_add_executable(study-workflow
                  COMPONENT_NAME its-offset
                  SOURCES src/its-offset-study-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingStudy)

o2_add_executable(dump-workfow
                  COMPONENT_NAME bc-tracks
                  SOURCES src/track-dump-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingStudy)
