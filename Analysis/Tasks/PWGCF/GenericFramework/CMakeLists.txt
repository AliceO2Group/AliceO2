# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(GenericFramework
               SOURCES  src/GFWCumulant.cxx
                        src/GFW.cxx
                        src/ProfileSubset.cxx
                        src/FlowContainer.cxx
                        src/GFWWeights.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisDataModel)

o2_target_root_dictionary(GenericFramework
              HEADERS include/GenericFramework/GFWCumulant.h
                      include/GenericFramework/GFW.h
                      include/GenericFramework/ProfileSubset.h
                      include/GenericFramework/FlowContainer.h
                      include/GenericFramework/GFWWeights.h
              LINKDEF src/GenericFrameworkLinkDef.h)

o2_add_dpl_workflow(flow-generic-framework
                    SOURCES GenericFramework.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::AnalysisCore O2::GenericFramework
                    COMPONENT_NAME Analysis) 