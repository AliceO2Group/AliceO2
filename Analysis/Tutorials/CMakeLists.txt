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
                    COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(track-collection-iteration
                  SOURCES src/trackCollectionIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(track-iteration
                  SOURCES src/trackIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(full-track-iteration
                  SOURCES src/fullTrackIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(collision-tracks-iteration
                  SOURCES src/collisionTracksIteration.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(new-collections
                  SOURCES src/newCollections.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(associated-derived
                  SOURCES src/associatedExample.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(dynamic-columns
                  SOURCES src/dynamicColumns.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histograms
                  SOURCES src/histograms.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histograms-full-tracks
                  SOURCES src/histogramsFullTracks.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(filters
                  SOURCES src/filters.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(partitions
                  SOURCES src/partitions.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(aodwriter
                  SOURCES src/aodwriter.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(aodreader
                  SOURCES src/aodreader.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(outputs
                  SOURCES src/outputs.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(jet-analysis
                  SOURCES src/jetAnalysis.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(tracks-combinations
                  SOURCES src/tracksCombinations.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(event-mixing
                  SOURCES src/eventMixing.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(mc-histograms
                  SOURCES src/mcHistograms.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(ccdbaccess
                  SOURCES src/ccdbaccess.cxx
                  PUBLIC_LINK_LIBRARIES O2::AnalysisCore O2::CCDB O2::Framework O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(weak-decay-iteration
                  SOURCES src/weakDecayIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(extended-columns
                  SOURCES src/extendedColumns.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(zdc-vzero-iteration
                  SOURCES src/ZDCVZeroIteration.cxx
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(hist-helpers-test
                    SOURCES src/histHelpersTest.cxx
                    PUBLIC_LINK_LIBRARIES O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(histogram-registry
                  SOURCES src/histogramRegistry.cxx
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
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::ConfigurableCut
                    COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(muon-iteration
                  SOURCES src/muonIteration.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(schema-evolution
                  SOURCES src/schemaEvolution.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(efficiency-per-run
                  SOURCES src/efficiencyPerRun.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(efficiency-global
                  SOURCES src/efficiencyGlobal.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(compatible-bcs
                  SOURCES src/compatibleBCs.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)

o2_add_dpl_workflow(multiplicity-event-track-selection
                  SOURCES src/multiplicityEventTrackSelection.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::AnalysisDataModel
                  COMPONENT_NAME AnalysisTutorial)
