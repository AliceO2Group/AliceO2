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

o2_add_executable(output-wildcard-workflow
                  SOURCES src/o2OutputWildcardWorkflow.cxx
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

# FIXME: given its name, should this one be a test instead of an executable ?
o2_add_executable(test_MakeDPLObjects
                  SOURCES test/test_MakeDPLObjects.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

# FIXME: given its name, should this one be a test instead of an executable ?
o2_add_executable(test_RawDeviceInjector
                  SOURCES src/test_RawDeviceInjector.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

# FIXME: given its name, should this one be a test instead of an executable ?
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

o2_add_executable(ccdb-fetch-to-timeframe
                  SOURCES src/test_CCDBFetchToTimeframe.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME TestWorkflows)

# Detector specific dummy workflows
o2_add_executable(tof-dummy-ccdb
                  SOURCES src/tof-dummy-ccdb.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::TOFReconstruction
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
