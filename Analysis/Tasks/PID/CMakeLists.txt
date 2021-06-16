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

# TOF

o2_add_dpl_workflow(pid-tof
                    SOURCES pidTOF.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(pid-tof-beta
                    SOURCES pidTOFbeta.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(pid-tof-qa-mc
                    SOURCES qaTOFMC.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(pid-tof-full
                    SOURCES pidTOFFull.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

# TPC

o2_add_dpl_workflow(pid-tpc
                    SOURCES pidTPC.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

o2_add_dpl_workflow(pid-tpc-full
                    SOURCES pidTPCFull.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)

# HMPID

o2_add_dpl_workflow(pid-hmpid-qa
                    SOURCES qaHMPID.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsBase O2::AnalysisDataModel O2::AnalysisCore
                    COMPONENT_NAME Analysis)
