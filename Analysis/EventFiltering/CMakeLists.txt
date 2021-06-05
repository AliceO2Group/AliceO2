# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EventFiltering
               SOURCES centralEventFilterProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore)

o2_add_executable(central-event-filter-processor
                  SOURCES cefp.cxx
                  COMPONENT_NAME Analysis
                  PUBLIC_LINK_LIBRARIES O2::EventFiltering)

o2_add_dpl_workflow(nuclei-filter
                    SOURCES nucleiFilter.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_library(CutHolder
               SOURCES PWGUD/cutHolder.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore)

o2_target_root_dictionary(CutHolder
                          HEADERS PWGUD/cutHolder.h
                          LINKDEF PWGUD/CutHolderLinkDef.h)

o2_add_dpl_workflow(diffraction-filter
                    SOURCES PWGUD/diffractionFilter.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore O2::CutHolder
                    COMPONENT_NAME Analysis)
