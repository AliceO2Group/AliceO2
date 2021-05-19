# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_dpl_workflow(
  tpcspectra-task-skim-reference
  SOURCES spectraTPCReference.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  tpcspectra-task-skim-provider
  SOURCES spectraTPCProvider.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  tpcspectra-task-skim-analyser
  SOURCES spectraTPCAnalyser.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  nucleispectra-task-skim-reference
  SOURCES spectraNucleiReference.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  nucleispectra-task-skim-provider
  SOURCES spectraNucleiProvider.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  nucleispectra-task-skim-analyser
  SOURCES spectraNucleiAnalyser.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)
if(FastJet_FOUND)
  o2_add_dpl_workflow(
    jet-task-skim-provider
    SOURCES jetProvider.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::DetectorsBase
      O2::AnalysisDataModel
      O2::AnalysisCore
      O2::AnalysisJets
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jetspectra-task-skim-analyser
    SOURCES jetSpectraAnalyser.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::DetectorsBase
      O2::AnalysisDataModel
      O2::AnalysisCore
      O2::AnalysisJets
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jetspectra-task-skim-reference
    SOURCES jetSpectraReference.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::DetectorsBase
      O2::AnalysisDataModel
      O2::AnalysisCore
      O2::AnalysisJets
    COMPONENT_NAME Analysis)
endif()

o2_add_dpl_workflow(
  upcspectra-task-skim-reference
  SOURCES spectraUPCReference.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  upcspectra-task-skim-provider
  SOURCES spectraUPCProvider.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)

o2_add_dpl_workflow(
  upcspectra-task-skim-analyser
  SOURCES spectraUPCAnalyser.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DetectorsBase
    O2::AnalysisDataModel
    O2::AnalysisCore
  COMPONENT_NAME Analysis)
