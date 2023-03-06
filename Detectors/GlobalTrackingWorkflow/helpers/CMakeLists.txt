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

o2_add_library(GlobalTrackingWorkflowHelpers
               SOURCES src/InputHelper.cxx
                       src/NoInpDummyOutSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                     O2::Framework
               PRIVATE_LINK_LIBRARIES O2::GlobalTracking # We link all the workflow libs etc as PRIVATE, such that we do not expose tons of headers automatically
                                      O2::ITSWorkflow
                                      O2::ITSMFTWorkflow
                                      O2::MCHIO
                                      O2::MIDWorkflow
                                      O2::MFTWorkflow
                                      O2::TPCWorkflow
                                      O2::GlobalTrackingWorkflowReaders
                                      O2::TOFWorkflowIO
                                      O2::FT0Workflow
                                      O2::FV0Workflow
                                      O2::FDDWorkflow
                                      O2::ZDCWorkflow
                                      O2::ITSMFTWorkflow
                                      O2::TRDWorkflowIO
                                      O2::CTPWorkflowIO
                                      O2::TPCReaderWorkflow
                                      O2::PHOSWorkflow
                                      O2::CPVWorkflow
                                      O2::EMCALWorkflow
                                      O2::SimulationDataFormat)

o2_add_executable(track-cluster-reader
        COMPONENT_NAME global
        TARGETVARNAME targetName
        SOURCES src/GlobalTrackClusterReader.cxx
        PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflowHelpers O2::DetectorsRaw)
