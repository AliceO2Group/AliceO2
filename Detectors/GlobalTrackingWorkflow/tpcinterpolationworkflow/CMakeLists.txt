# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

#TODO Does the O2::GlobalTracking library need to be linked?
o2_add_library(TPCInterpolationWorkflow
               SOURCES src/TPCInterpolationSpec.cxx
                       src/TPCResidualWriterSpec.cxx
                       src/TrackInterpolationReaderSpec.cxx
                       src/TrackInterpolationWorkflow.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTracking
                                     O2::ITSWorkflow
                                     O2::SpacePoints
                                     O2::GlobalTrackingWorkflow
                                     O2::TOFWorkflowIO
                                     O2::Framework
                                     )

o2_add_executable(scdcalib-interpolation-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-interpolation-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCInterpolationWorkflow)
