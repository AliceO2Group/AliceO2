# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See https://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "Workflow")
set(MODULE_BUCKET_NAME QC_workflow_bucket)

O2_SETUP(NAME ${MODULE_NAME})
#set(SRCS
#   )

## TODO: feature of macro, it deletes the variables we pass to it, set them again
## this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

#O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME simple-merger-workflow

  SOURCES
  src/SimpleMergerWorkflow.cxx
  src/RootObjectProducerSpec.cxx
  src/RootObjectMergerSpec.cxx

  BUCKET_NAME ${MODULE_BUCKET_NAME}
)
