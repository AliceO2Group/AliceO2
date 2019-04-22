# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "GlobalTrackingWorkflow")
set(MODULE_BUCKET_NAME GlobalTracking_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/TrackWriterTPCITSSpec.cxx
  src/TPCITSMatchingSpec.cxx
  src/MatchTPCITSWorkflow.cxx
  )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME tpcits-match-workflow
  
  SOURCES
  src/tpcits-match-workflow.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  )
