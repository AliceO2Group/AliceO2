# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_dpl_workflow(table-maker
                    SOURCES tableMaker.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(table-reader
                    SOURCES tableReader.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(dilepton-ee
                    SOURCES dileptonEE.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(dilepton-mumu
                    SOURCES dileptonMuMu.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(dq-filter-pp
                    SOURCES filterPP.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)                    

o2_add_dpl_workflow(table-maker-muon-pp
                    SOURCES tableMakerMuon_pp.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::DetectorsBase O2::AnalysisCore O2::PWGDQCore
                    COMPONENT_NAME Analysis)
