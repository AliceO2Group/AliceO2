# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "DPLUtils")
set(MODULE_BUCKET_NAME DPLUtils_bucket)

O2_SETUP(NAME ${MODULE_NAME})
set(SRCS
        src/Utils.cxx
        src/DPLBroadcaster.cxx
        src/DPLGatherer.cxx
        src/DPLMerger.cxx
        src/DPLRouter.cxx
        test/DPLBroadcasterMerger.cxx
        test/DPLOutputTest.cxx
        )

set(HEADERS
        include/${MODULE_NAME}/Utils.h
        include/${MODULE_NAME}/RootTreeReader.h
        include/${MODULE_NAME}/RootTreeWriter.h
        include/${MODULE_NAME}/MakeRootTreeWriterSpec.h
        )

## TODO: feature of macro, it deletes the variables we pass to it, set them again
## this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_FRAMEWORK_WORKFLOW(
        WORKFLOW_NAME "test_DPLBroadcasterMerger"
        DETECTOR_BUCKETS DPLUtils_bucket
        SOURCES src/Utils.cxx test/test_DPLBroadcasterMerger.cxx test/DPLBroadcasterMerger.cxx src/DPLMerger.cxx src/DPLBroadcaster.cxx
)

O2_FRAMEWORK_WORKFLOW(
        WORKFLOW_NAME "test_DPLOutputTest"
        DETECTOR_BUCKETS DPLUtils_bucket
        SOURCES src/Utils.cxx test/test_DPLOutputTest.cxx test/DPLOutputTest.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS
  test/test_RootTreeWriter.cxx

  TIMEOUT 60
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS
  test/test_RootTreeReader.cxx
  test/test_RootTreeWriterWorkflow.cxx

  TIMEOUT 60
  COMMAND_LINE_ARGS
  --run
)
