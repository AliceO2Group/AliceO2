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

o2_add_library(ITSCAWorkflow
               TARGETVARNAME targetName
               SOURCES src/ConfigPreflight.cxx
                       src/CATrackerSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimulationDataFormat
                                     O2::DataFormatsITS
                                     O2::DataFormatsITSMFT
                                     O2::ITSBase
                                     O2::ITSMFTTracking
                                     O2::ITSMFTCAWriter
                                     O2::MFTTracking
                                     O2::Steer
                                     O2::CCDB)

o2_add_executable(ca-tracker-workflow
                  SOURCES src/its-ca-tracker-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSCAWorkflow)

o2_add_test(its-ca-config-preflight
            COMPONENT_NAME its
            LABELS "its;workflow;itsmft"
            SOURCES test/testITSCAConfigPreflight.cxx
            PUBLIC_LINK_LIBRARIES O2::ITSCAWorkflow)

o2_add_test(its-ca-tracker-dpl-contract
            COMPONENT_NAME its
            LABELS "its;workflow;itsmft"
            SOURCES test/testITSCATrackerDPLContract.cxx
            PUBLIC_LINK_LIBRARIES O2::ITSCAWorkflow)

o2_add_test(its-ca-truth-seeding
            COMPONENT_NAME its
            LABELS "its;workflow;itsmft"
            SOURCES test/testITSCATruthSeeding.cxx
            PUBLIC_LINK_LIBRARIES O2::ITSCAWorkflow)
