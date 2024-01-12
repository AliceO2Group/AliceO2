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

o2_add_dpl_workflow(dummy-workflow
                  SOURCES src/o2DummyWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(detect-missing-timeframe
                  SOURCES test/test_DetectMissingTimeframe.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(wait-until-possible
                  SOURCES test/test_WaitUntilPossible.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(o2rootmessage-workflow
                  SOURCES "src/test_o2RootMessageWorkflow.cxx"
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(diamond-workflow
                  SOURCES src/o2DiamondWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(diamond-workflow-leaky
                  SOURCES src/o2DiamondWorkflowLeaky.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(output-wildcard-workflow
                  SOURCES src/o2OutputWildcardWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(parallel-workflow
                  SOURCES src/o2ParallelWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(sync-reconstruction-dummy
                  SOURCES src/o2SyncReconstructionDummy.cxx
                  COMPONENT_NAME TestWorkflows)

# FIXME: renable when the new ROOT is used
#o2_add_dpl_workflow(d0-analysis
#                  SOURCES src/o2D0Analysis.cxx
#                  PUBLIC_LINK_LIBRARIES O2::FrameworkRootAnalysisHelpers
#                  COMPONENT_NAME TestWorkflows)
#
#o2_add_dpl_workflow(simple-tracks-analysis
#                  SOURCES src/o2SimpleTracksAnalysis.cxx
#                  PUBLIC_LINK_LIBRARIES O2::FrameworkRootAnalysisHelpers
#                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(analysis-task-example
                  SOURCES src/o2AnalysisTaskExample.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(data-query-workflow
                  SOURCES src/o2DataQueryWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

# FIXME: given its name, should this one be a test instead of an executable ?
o2_add_dpl_workflow(test_MakeDPLObjects
                  SOURCES test/test_MakeDPLObjects.cxx
                  COMPONENT_NAME TestWorkflows)

# FIXME: given its name, should this one be a test instead of an executable ?
o2_add_dpl_workflow(test_RawDeviceInjector
                  SOURCES src/test_RawDeviceInjector.cxx
                  COMPONENT_NAME TestWorkflows)

# FIXME: given its name, should this one be a test instead of an executable ?
o2_add_dpl_workflow(test_CompletionPolicies
                  SOURCES src/test_CompletionPolicies.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(ccdb-fetch-to-timeframe
                  SOURCES src/test_CCDBFetchToTimeframe.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(simple-source
                  SOURCES src/o2SimpleSource.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(simple-fairmq-source
                  SOURCES src/o2SimpleFairMQSource.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(simple-fairmq-sink
                  SOURCES src/o2SimpleFairMQSink.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(simple-sink
                  SOURCES src/o2SimpleSink.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(analysis-workflow
                  SOURCES src/o2AnalysisWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

# Detector specific dummy workflows
o2_add_dpl_workflow(tof-dummy-ccdb
                  SOURCES src/tof-dummy-ccdb.cxx
                  PUBLIC_LINK_LIBRARIES O2::TOFReconstruction
                  COMPONENT_NAME TestWorkflows)

# Detector specific dummy workflows
o2_add_dpl_workflow(test-ccdb-fetcher
                    SOURCES src/test_CCDBFetcher.cxx
                    PUBLIC_LINK_LIBRARIES O2::DataFormatsTOF O2::Framework
                    COMPONENT_NAME TestWorkflows)

add_executable(o2-test-deadlock src/o2DeadlockReproducer.cxx)
target_link_libraries(o2-test-deadlock PUBLIC FairMQ::FairMQ)
add_test(NAME o2-test-deadlock COMMAND o2-test-deadlock --channel-config "name=data,type=sub,method=bind,address=ipc://127.0.0.1,rateLogging=0" --id foo --transport zeromq --control static)
