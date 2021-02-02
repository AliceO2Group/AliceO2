# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_dpl_workflow(qatask
                    SOURCES qaTask.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(qa-efficiency
                    SOURCES qaTaskEfficiency.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(qa-simple
                    SOURCES qaTaskSimple.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-track-quality-selector
                    SOURCES HFTrackQualitySelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-track-index-skims-creator
                    SOURCES HFTrackIndexSkimsCreator.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-creator-2prong
                    SOURCES HFCandidateCreator2Prong.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-creator-3prong
                    SOURCES HFCandidateCreator3Prong.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-selector-d0topik
                    SOURCES HFCandidateSelectorD0ToPiK.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-selector-lctopkpi
                    SOURCES HFCandidateSelectorLcToPKPi.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-selector-jpsitoee  
                    SOURCES HFCandidateSelectorJpsiToEE.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-d0
                    SOURCES taskD0.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-dplus
                    SOURCES taskDPlus.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-lc
                    SOURCES taskLc.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-jpsi                                
                    SOURCES taskJpsi.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)
