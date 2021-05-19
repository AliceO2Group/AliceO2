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

o2_add_dpl_workflow(histogram-track-selection
                    SOURCES src/histogramTrackSelection.cxx
                    PUBLIC_LINK_LIBRARIES O2::AnalysisDataModel O2::AnalysisCore
                    JOB_POOL analysis
                    COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(track-collection-iteration
                  SOURCES src/trackCollectionIteration.cxx
                  JOB_POOL analysis
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(track-iteration
                  SOURCES src/trackIteration.cxx
                  JOB_POOL analysis
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(full-track-iteration
                  SOURCES src/fullTrackIteration.cxx
                  JOB_POOL analysis
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(collision-tracks-iteration
                  SOURCES src/collisionTracksIteration.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(new-collections
                  SOURCES src/newCollections.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(associated-derived
                  SOURCES src/associatedExample.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(dynamic-columns
                  SOURCES src/dynamicColumns.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histograms
                  SOURCES src/histograms.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histograms-full-tracks
                  SOURCES src/histogramsFullTracks.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(filters
                  SOURCES src/filters.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(partitions
                  SOURCES src/partitions.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(aodwriter
                  SOURCES src/aodwriter.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(aodreader
                  SOURCES src/aodreader.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(tableio-write
                  SOURCES src/tableIOout.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(tableio-read
                  SOURCES src/tableIOin.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(outputs
                  SOURCES src/outputs.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(jet-analysis
                  SOURCES src/jetAnalysis.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(tracks-combinations
                  SOURCES src/tracksCombinations.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(event-mixing
                  SOURCES src/eventMixing.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(mc-histograms
                  SOURCES src/mcHistograms.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(ccdbaccess
                  SOURCES src/ccdbaccess.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::AnalysisCore O2::CCDB O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(weak-decay-iteration
                  JOB_POOL analysis
                  SOURCES src/weakDecayIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(extended-columns
                  SOURCES src/extendedColumns.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(extended-tables
                  SOURCES src/extendedTables.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(zdc-vzero-iteration
                  SOURCES src/ZDCVZeroIteration.cxx
                  JOB_POOL analysis
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histogram-registry
                  SOURCES src/histogramRegistry.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_library(ConfigurableCut
               SOURCES  src/configurableCut.cxx
               PUBLIC_LINK_LIBRARIES O2::AnalysisCore)

o2_target_root_dictionary(ConfigurableCut
                          HEADERS include/Analysis/configurableCut.h
                          LINKDEF src/ConfigurableCutLinkDef.h)

o2_add_dpl_workflow(configurable-objects
                    SOURCES src/configurableObjects.cxx
                    JOB_POOL analysis
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::ConfigurableCut
                    COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(muon-iteration
                  SOURCES src/muonIteration.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(schema-evolution
                  SOURCES src/schemaEvolution.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(efficiency-per-run
                  SOURCES src/efficiencyPerRun.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(efficiency-global
                  SOURCES src/efficiencyGlobal.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(compatible-bcs
                  SOURCES src/compatibleBCs.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(multiplicity-event-track-selection
                  SOURCES src/multiplicityEventTrackSelection.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(multiprocess-example
                  SOURCES src/multiProcess.cxx
                  JOB_POOL analysis
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)
