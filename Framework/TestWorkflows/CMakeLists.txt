# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# FIXME Is this one supposed to be a header only library (in which case the .h
# to be installed should be in include/TestWorkflows) or not a library at all ?
# o2_add_library(TestWorkflows SOURCES src/dummy.cxx
#
# set(HEADERS src/o2_sim_its_ALP3.h src/o2_sim_tpc.h )
#

o2_add_executable(dummy-workflow
                  SOURCES src/o2DummyWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(o2rootmessage-workflow
                  SOURCES "src/test_o2RootMessageWorkflow.cxx"
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(diamond-workflow
                  SOURCES src/o2DiamondWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(parallel-workflow
                  SOURCES src/o2ParallelWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(flp-qualification
                  SOURCES src/flpQualification.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(sync-reconstruction-dummy
                  SOURCES src/o2SyncReconstructionDummy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(aod-dummy-workflow
                  SOURCES src/o2AODDummy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(d0-analysis
                  SOURCES src/o2D0Analysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(simple-tracks-analysis
                  SOURCES src/o2SimpleTracksAnalysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(analysis-task-example
                  SOURCES src/o2AnalysisTaskExample.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(data-query-workflow
                  SOURCES src/o2DataQueryWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(test_MakeDPLObjects
                  SOURCES test/test_MakeDPLObjects.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(test_RawDeviceInjector
                  SOURCES src/test_RawDeviceInjector.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(test_CompletionPolicies
                  SOURCES src/test_CompletionPolicies.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(datasampling-pod-and-root
                  SOURCES src/dataSamplingPodAndRoot.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(datasampling-parallel
                  SOURCES src/dataSamplingParallel.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

o2_add_executable(datasampling-time-pipeline
                  SOURCES src/dataSamplingTimePipeline.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)
#
# FIXME: reconnect those when simulation is enabled
#
# # These should be enabled only if one uses the full O2 default if
# (PYTHIA8_INCLUDE_DIR) O2_FRAMEWORK_WORKFLOW( WORKFLOW_NAME "o2ITSClusterizers"
# DETECTOR_BUCKETS its_simulation_bucket its_reconstruction_bucket
# passive_detector_bucket generators_base_bucket generators_bucket run_bucket
# SOURCES src/test_o2ITSCluserizer.cxx src/o2_sim_its_ALP3.cxx )
#
# O2_FRAMEWORK_WORKFLOW( WORKFLOW_NAME "o2TPCSimulation" DETECTOR_BUCKETS
# tpc_simulation_bucket tpc_reconstruction_bucket passive_detector_bucket
# generators_base_bucket generators_bucket run_bucket SOURCES
# src/test_o2TPCSimulation.cxx src/o2_sim_tpc.cxx ) ENDIF(PYTHIA8_INCLUDE_DIR)
