# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# FIXME: do we actually need a library here, or is the executable enough ?

o2_add_library(GlobalTrackingWorkflow
               SOURCES src/TrackWriterTPCITSSpec.cxx
                       src/TPCITSMatchingSpec.cxx
                       src/PrimaryVertexingSpec.cxx
                       src/PrimaryVertexWriterSpec.cxx
                       src/VertexTrackMatcherSpec.cxx
                       src/SecondaryVertexingSpec.cxx
                       src/SecondaryVertexWriterSpec.cxx
                       src/CosmicsMatchingSpec.cxx
                       src/TrackCosmicsWriterSpec.cxx
                       src/TOFMatcherSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTracking
                                     O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflowHelpers
                                     O2::ITStracking
                                     O2::ITSWorkflow
                                     O2::MFTTracking
                                     O2::MFTWorkflow
                                     O2::TPCWorkflow
                                     O2::FT0Workflow
                                     O2::ITSMFTWorkflow
                                     O2::SimulationDataFormat
                                     O2::DetectorsVertexing)

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

o2_add_executable(matcher-workflow
                  COMPONENT_NAME tof
                  SOURCES src/tof-matcher-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow O2::TOFWorkflowIO)


add_subdirectory(tofworkflow)
add_subdirectory(tpcinterpolationworkflow)
add_subdirectory(helpers)
add_subdirectory(readers)
