# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

add_subdirectory(PWGCF)
add_subdirectory(PWGPP)
add_subdirectory(PWGDQ)
add_subdirectory(PWGHF)
add_subdirectory(PWGJE)
add_subdirectory(PWGLF)
add_subdirectory(PWGUD)
add_subdirectory(Utils)
add_subdirectory(SkimmingTutorials)
add_subdirectory(PID)

o2_add_dpl_workflow(trackextension
                    SOURCES trackextension.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(trackselection
                    SOURCES trackselection.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(trackqa
                    SOURCES trackqa.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(validation
                    SOURCES validation.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(event-selection
                    SOURCES eventSelection.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase O2::CCDB O2::CommonConstants
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(event-selection-qa
                    SOURCES eventSelectionQa.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(multiplicity-table
                    SOURCES multiplicityTable.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(multiplicity-qa
                    SOURCES multiplicityQa.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(centrality-table
                    SOURCES centralityTable.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase O2::CCDB
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(centrality-qa
                    SOURCES centralityQa.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(timestamp
                    SOURCES timestamp.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::DetectorsRaw O2::AnalysisCore O2::CommonDataFormat O2::CCDB
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(weak-decay-indices
                    SOURCES weakDecayIndices.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(emcal-correction-task
                    SOURCES emcalCorrectionTask.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::DetectorsBase O2::EMCALBase O2::EMCALReconstruction
                    COMPONENT_NAME Analysis)
