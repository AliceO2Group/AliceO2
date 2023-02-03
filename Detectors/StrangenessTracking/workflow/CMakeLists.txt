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

o2_add_library(StrangenessTrackingWorkflow
               TARGETVARNAME targetName
               SOURCES src/StrangenessTrackingSpec.cxx
               SOURCES src/StrangenessTrackingWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflow
                                     O2::ITSWorkflow
                                     O2::StrangenessTracking
                                     O2::SimulationDataFormat
                                     O2::ReconstructionDataFormats
                                     O2::Framework
                                     O2::DetectorsRaw
                                     O2::SimConfig
                                     O2::DataFormatsITS
                                     O2::DataFormatsDCS
                                     O2::ITStracking
                                     O2::ITSReconstruction
                                     O2::ITSMFTReconstruction
                                     O2::ITSMFTWorkflow
                                     O2::DetectorsCalibration
                                     O2::GlobalTrackingWorkflowWriters
                                     O2::DCAFitter
                                     O2::CCDB)

o2_add_executable(strangeness-tracking-workflow
                  SOURCES src/strangeness-tracking-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::StrangenessTrackingWorkflow
                                        O2::GlobalTrackingWorkflowReaders)
