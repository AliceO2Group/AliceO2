# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See https://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE_NAME "TestWorkflows")
set(MODULE_BUCKET_NAME FrameworkApplication_bucket)

O2_SETUP(NAME ${MODULE_NAME})
set(SRCS
    src/dummy.cxx
   )

## TODO: feature of macro, it deletes the variables we pass to it, set them again
## this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
  EXE_NAME "o2DummyWorkflow"
  SOURCES "src/o2DummyWorkflow.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "test_o2RootMessageWorkflow"
  SOURCES "src/test_o2RootMessageWorkflow.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "o2DiamondWorkflow"
  SOURCES "src/o2DiamondWorkflow.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "o2ParallelWorkflow"
  SOURCES "src/o2ParallelWorkflow.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "test_MakeDPLObjects"
  SOURCES "test/test_MakeDPLObjects.cxx"
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

O2_GENERATE_EXECUTABLE(
  EXE_NAME "test_RawDeviceInjector"
  SOURCES src/test_RawDeviceInjector.cxx
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

# These should be enabled only if one uses the full O2 default
if (PYTHIA8_INCLUDE_DIR)
  O2_FRAMEWORK_WORKFLOW(
    WORKFLOW_NAME "o2ITSClusterizers"
    DETECTOR_BUCKETS its_simulation_bucket
                     its_reconstruction_bucket
                     passive_detector_bucket
                     generators_base_bucket
                     generators_bucket
                     run_bucket
    SOURCES src/test_o2ITSCluserizer.cxx src/o2_sim_its_ALP3.cxx
  )

  O2_FRAMEWORK_WORKFLOW(
    WORKFLOW_NAME "o2TPCSimulation"
    DETECTOR_BUCKETS tpc_simulation_bucket
                     tpc_reconstruction_bucket
                     passive_detector_bucket
                     generators_base_bucket
                     generators_bucket
                     run_bucket
    SOURCES src/test_o2TPCSimulation.cxx src/o2_sim_tpc.cxx
  )
ENDIF(PYTHIA8_INCLUDE_DIR)
