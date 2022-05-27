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

o2_add_library(MFTWorkflow
               TARGETVARNAME targetName
               SOURCES src/RecoWorkflow.cxx
                       src/ClustererSpec.cxx
                       src/ClusterWriterSpec.cxx
                       src/TrackerSpec.cxx
                       src/TrackReaderSpec.cxx
                       src/TrackWriterSpec.cxx
                       src/MFTAssessmentSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::SimConfig
                                     O2::SimulationDataFormat
                                     O2::ITSMFTReconstruction
                                     O2::MFTTracking
                                     O2::MFTAssessment
                                     O2::DataFormatsMFT
                                     O2::ITSMFTWorkflow)
o2_add_executable(reco-workflow
                  SOURCES src/mft-reco-workflow.cxx
                  COMPONENT_NAME mft
                  PUBLIC_LINK_LIBRARIES O2::MFTWorkflow)

o2_add_executable(cluster-reader-workflow
                  SOURCES src/mft-cluster-reader-workflow.cxx
                  COMPONENT_NAME mft
                  PUBLIC_LINK_LIBRARIES O2::MFTWorkflow)

o2_add_executable(assessment-workflow
                  SOURCES src/mft-assessment-workflow.cxx
                  COMPONENT_NAME mft
                  PUBLIC_LINK_LIBRARIES O2::MFTWorkflow)

o2_add_executable(cluster-writer-workflow
                  SOURCES src/mft-cluster-writer-workflow.cxx
                  COMPONENT_NAME mft
                  PUBLIC_LINK_LIBRARIES O2::MFTWorkflow)
