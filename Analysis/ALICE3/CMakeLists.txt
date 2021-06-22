# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_header_only_library(ALICE3Analysis)

o2_add_dpl_workflow(alice3-trackselection
                    SOURCES src/alice3-trackselection.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(alice3-trackextension
                    SOURCES src/alice3-trackextension.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(alice3-qa-multiplicity
                    SOURCES src/alice3-qa-multiplicity.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(alice3-pid-rich-qa
                    SOURCES src/pidRICHqa.cxx
                    PUBLIC_LINK_LIBRARIES O2::AnalysisDataModel O2::AnalysisCore O2::ALICE3Analysis
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(alice3-pid-ftof-qa
                    SOURCES src/pidFTOFqa.cxx
                    PUBLIC_LINK_LIBRARIES O2::AnalysisDataModel O2::AnalysisCore O2::ALICE3Analysis
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(alice3-pid-tof
                    SOURCES src/alice3-pidTOF.cxx
                    PUBLIC_LINK_LIBRARIES O2::AnalysisDataModel O2::AnalysisCore O2::ALICE3Analysis
                    COMPONENT_NAME Analysis)
