# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
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

o2_add_dpl_workflow(dummy-workflow
                  SOURCES src/o2DummyWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(o2rootmessage-workflow
                  SOURCES "src/test_o2RootMessageWorkflow.cxx"
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(diamond-workflow
                  SOURCES src/o2DiamondWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(output-wildcard-workflow
                  SOURCES src/o2OutputWildcardWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(parallel-workflow
                  SOURCES src/o2ParallelWorkflow.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(flp-qualification
                  SOURCES src/flpQualification.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(sync-reconstruction-dummy
                  SOURCES src/o2SyncReconstructionDummy.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(d0-analysis
                  SOURCES src/o2D0Analysis.cxx
                  COMPONENT_NAME TestWorkflows)

o2_add_dpl_workflow(simple-tracks-analysis
                  SOURCES src/o2SimpleTracksAnalysis.cxx
                  COMPONENT_NAME TestWorkflows)

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

o2_add_dpl_workflow(simple-sink
                  SOURCES src/o2SimpleSink.cxx
                  COMPONENT_NAME TestWorkflows)

# Detector specific dummy workflows
o2_add_dpl_workflow(tof-dummy-ccdb
                  SOURCES src/tof-dummy-ccdb.cxx
                  PUBLIC_LINK_LIBRARIES O2::TOFReconstruction
                  COMPONENT_NAME TestWorkflows)

if(BUILD_SIMULATION)
  o2_add_executable(
    ITSClusterizers
    COMPONENT_NAME TestWorkflows
    PUBLIC_LINK_LIBRARIES O2::ITSSimulation O2::ITSReconstruction
                          O2::DetectorsPassive O2::Generators O2::Framework
    SOURCES src/test_o2ITSCluserizer.cxx src/o2_sim_its_ALP3.cxx)

  o2_add_executable(TPCSimulation
                    COMPONENT_NAME TestWorkflows
                    PUBLIC_LINK_LIBRARIES O2::TPCSimulation
                                          O2::TPCReconstruction
                                          O2::DetectorsPassive O2::Generators
                                          O2::Framework
                    SOURCES src/test_o2TPCSimulation.cxx src/o2_sim_tpc.cxx)
endif()
