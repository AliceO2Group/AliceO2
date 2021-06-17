# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_dpl_workflow(hf-track-index-skims-creator
                    SOURCES HFTrackIndexSkimsCreator.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing O2::AnalysisTasksUtils ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-creator-2prong
                    SOURCES HFCandidateCreator2Prong.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-tree-creator-d0-tokpi
                    SOURCES HFTreeCreatorD0ToKPi.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-creator-cascade
                    SOURCES HFCandidateCreatorCascade.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing O2::AnalysisTasksUtils ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-candidate-creator-3prong
                    SOURCES HFCandidateCreator3Prong.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-tree-creator-lc-topkpi
                    SOURCES HFTreeCreatorLcToPKPi.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing ROOT::EG
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-d0-candidate-selector
                    SOURCES HFD0CandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-dplus-topikpi-candidate-selector
                    SOURCES HFDplusToPiKPiCandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-lc-candidate-selector
                    SOURCES HFLcCandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-jpsi-toee-candidate-selector
                    SOURCES HFJpsiToEECandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-xic-topkpi-candidate-selector
                    SOURCES HFXicToPKPiCandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-lc-tok0sp-candidate-selector
                    SOURCES HFLcK0sPCandidateSelector.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing O2::AnalysisTasksUtils
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

o2_add_dpl_workflow(hf-task-bplus
                    SOURCES taskBPlus.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-xic
                    SOURCES taskXic.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-x
                    SOURCES taskX.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-mc-validation
                    SOURCES HFMCValidation.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-lc-tok0sp
                    SOURCES taskLcK0sP.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing O2::AnalysisTasksUtils
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-correlator-d0d0bar
                    SOURCES HFCorrelatorD0D0bar.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-correlator-dplusdminus
                    SOURCES HFCorrelatorDplusDminus.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(hf-task-correlation-ddbar
                    SOURCES taskCorrelationDDbar.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsVertexing
                    COMPONENT_NAME Analysis)
