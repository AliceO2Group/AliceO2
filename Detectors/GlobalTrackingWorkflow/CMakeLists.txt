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

# FIXME: do we actually need a library here, or is the executable enough ?

# add_compile_options(-O0 -g -fPIC)

o2_add_library(GlobalTrackingWorkflow
               SOURCES src/TrackWriterTPCITSSpec.cxx
                       src/TPCITSMatchingSpec.cxx
                       src/PrimaryVertexingSpec.cxx
                       src/PrimaryVertexWriterSpec.cxx
                       src/VertexTrackMatcherSpec.cxx
                       src/SecondaryVertexingSpec.cxx
                       src/StrangenessTrackingSpec.cxx
                       src/GlobalFwdMatchingSpec.cxx
                       src/SecondaryVertexWriterSpec.cxx
                       src/StrangenessTrackingWriterSpec.cxx
                       src/CosmicsMatchingSpec.cxx
                       src/TrackCosmicsWriterSpec.cxx
                       src/TOFMatcherSpec.cxx
                       src/TOFMatchChecker.cxx
                       src/HMPMatcherSpec.cxx
                       src/TOFEventTimeChecker.cxx
                       src/GlobalFwdTrackWriterSpec.cxx
                       src/GlobalFwdMatchingAssessmentSpec.cxx
                       src/MatchedMFTMCHWriterSpec.cxx
                       src/ReaderDriverSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTracking
                                     O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflowHelpers
                                     O2::DataFormatsGlobalTracking
                                     O2::ITStracking
                                     O2::ITSWorkflow
                                     O2::MFTTracking
                                     O2::MFTWorkflow
                                     O2::HMPIDWorkflow
                                     O2::TPCWorkflow
                                     O2::FT0Workflow
                                     O2::ITSMFTWorkflow
                                     O2::SimulationDataFormat
                                     O2::DetectorsVertexing
                                     O2::StrangenessTracking)
o2_add_executable(driver-workflow
                  COMPONENT_NAME reader
                  SOURCES src/reader-driver-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow )

o2_add_executable(match-workflow
                  COMPONENT_NAME tpcits
                  SOURCES src/tpcits-match-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

o2_add_executable(match-workflow
                  COMPONENT_NAME cosmics
                  SOURCES src/cosmics-match-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow  O2::TOFWorkflowIO)

o2_add_executable(vertexing-workflow
                  COMPONENT_NAME primary
                  SOURCES src/primary-vertexing-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::TOFWorkflowIO)

o2_add_executable(vertex-reader-workflow
                  COMPONENT_NAME primary
                  SOURCES src/primary-vertex-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow )

o2_add_executable(vertex-reader-workflow
                  COMPONENT_NAME secondary
                  SOURCES src/secondary-vertex-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow )

o2_add_executable(vertexing-workflow
                  COMPONENT_NAME secondary
                  SOURCES src/secondary-vertexing-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow  O2::TOFWorkflowIO)

o2_add_executable(strangeness-tracking-workflow
                  SOURCES src/strangeness-tracking-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

o2_add_executable(matcher-workflow
                  COMPONENT_NAME tof
                  SOURCES src/tof-matcher-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::TOFWorkflowIO)
o2_add_executable(match-checker-workflow
                  COMPONENT_NAME tof
                  SOURCES src/tof-match-checker-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::TOFWorkflowIO)
o2_add_executable(match-eventtime-workflow
                  COMPONENT_NAME tof
                  SOURCES src/tof-eventtime-checker-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::TOFWorkflowIO)

o2_add_executable(matcher-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/hmp-matcher-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::HMPIDWorkflow)

o2_add_executable(matcher-workflow
                  COMPONENT_NAME globalfwd
                  SOURCES src/globalfwd-matcher-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

o2_add_executable(assessment-workflow
                  COMPONENT_NAME globalfwd
                  SOURCES src/globalfwd-assessment-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

o2_add_executable(writer-workflow
                  COMPONENT_NAME tfidinfo
                  SOURCES src/tfidinfo-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

add_subdirectory(tofworkflow)
add_subdirectory(tpcinterpolationworkflow)
add_subdirectory(helpers)
add_subdirectory(readers)
add_subdirectory(writers)
add_subdirectory(qc)

add_subdirectory(study)
