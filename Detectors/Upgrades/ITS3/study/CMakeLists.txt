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

# add_compile_options(-O0 -g -fPIC -fno-omit-frame-pointer)

o2_add_library(ITS3TrackingStudy
               TARGETVARNAME targetName
               SOURCES src/ITS3TrackingStudyParam.cxx
                       src/TrackingStudy.cxx
                       src/ParticleInfoExt.cxx
               PUBLIC_LINK_LIBRARIES O2::ITS3Workflow
                                     O2::GlobalTracking
                                     O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflowHelpers
                                     O2::DataFormatsGlobalTracking
                                     O2::DetectorsVertexing
                                     O2::SimulationDataFormat)

o2_target_root_dictionary(ITS3TrackingStudy
                          HEADERS include/ITS3TrackingStudy/ITS3TrackingStudyParam.h
                                  include/ITS3TrackingStudy/ParticleInfoExt.h
                          LINKDEF src/ITS3TrackingStudyLinkDef.h)

o2_add_executable(study-workflow
                  COMPONENT_NAME its3-tracking
                  SOURCES src/its3-tracking-study-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ITS3TrackingStudy)

add_subdirectory(macros)
