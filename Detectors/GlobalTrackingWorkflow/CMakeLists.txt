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
               SOURCES src/TrackWriterTPCITSSpec.cxx src/TPCITSMatchingSpec.cxx
                       src/MatchTPCITSWorkflow.cxx src/TrackTPCITSReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTracking O2::ITSWorkflow
                                     O2::TPCWorkflow O2::FITWorkflow
                                     O2::ITSMFTWorkflow)

o2_add_executable(match-workflow
                  COMPONENT_NAME tpcits
                  SOURCES src/tpcits-match-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflow)

add_subdirectory(tofworkflow)

