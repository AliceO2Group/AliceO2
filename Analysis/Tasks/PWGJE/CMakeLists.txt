# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(FastJet_FOUND)
  o2_add_dpl_workflow(
    jet-finder
    SOURCES jetfinder.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::AnalysisJets
      O2::AnalysisDataModel
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jet-finder-hf
    SOURCES jetfinderhf.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::AnalysisJets
      O2::AnalysisDataModel
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jet-finder-hadron-recoil
    SOURCES jetfinderhadronrecoil.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::AnalysisJets
      O2::AnalysisDataModel
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jet-substructure
    SOURCES jetsubstructure.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::AnalysisJets
      O2::AnalysisDataModel
    COMPONENT_NAME Analysis)

  o2_add_dpl_workflow(
    jet-skimmer
    SOURCES jetskimming.cxx
    PUBLIC_LINK_LIBRARIES
      O2::Framework
      O2::AnalysisJets
      O2::AnalysisDataModel
    COMPONENT_NAME Analysis)
endif()
