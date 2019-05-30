# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "TPCWorkflow")
set(MODULE_BUCKET_NAME TPC_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})
set(SRCS
   src/RecoWorkflow.cxx
   src/PublisherSpec.cxx
   src/ClustererSpec.cxx
   src/ClusterDecoderRawSpec.cxx
   src/CATrackerSpec.cxx
   #
   src/TrackReaderSpec.cxx
   )

if (OPENGL_FOUND AND GLFW_FOUND AND GLEW_FOUND AND OPENGL_GLU_FOUND AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    add_definitions(-DBUILD_EVENT_DISPLAY)
endif()

## TODO: feature of macro, it deletes the variables we pass to it, set them again
## this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME o2-tpc-reco-workflow

  SOURCES
  src/tpc-reco-workflow.cxx

  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

set(TEST_SRCS
      test/test_TPCWorkflow.cxx
   )

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
  TIMEOUT 60
)
