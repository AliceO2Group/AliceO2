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

add_library(AODProducerWorkflow INTERFACE)

target_link_libraries(
        AODProducerWorkflow
        INTERFACE
          O2::DetectorsVertexing
          O2::FT0Workflow
          O2::FDDWorkflow
          O2::FV0Workflow
          O2::Framework
          O2::GlobalTracking
          O2::GlobalTrackingWorkflow
          O2::ITSMFTWorkflow
          O2::ITSWorkflow
          O2::ITStracking
          O2::MCHTracking
          O2::MFTWorkflow
          O2::MathUtils
          O2::SimulationDataFormat
          O2::Steer
          O2::TPCWorkflow
          O2::ZDCBase
)

target_include_directories(AODProducerWorkflow INTERFACE include)

add_library(internal::AODProducerWorkflow ALIAS AODProducerWorkflow)

o2_add_executable(
  workflow
  COMPONENT_NAME aod-producer
  SOURCES src/aod-producer-workflow.cxx src/AODProducerWorkflowSpec.cxx
  PUBLIC_LINK_LIBRARIES internal::AODProducerWorkflow O2::Version
)

o2_add_executable(
  workflow
  COMPONENT_NAME aod-mc-producer
  SOURCES src/aod-mc-producer-workflow.cxx src/AODMcProducerWorkflowSpec.cxx
  PUBLIC_LINK_LIBRARIES internal::AODProducerWorkflow O2::Version
)

o2_add_executable(
        standalone-aod-producer
        COMPONENT_NAME reco
        SOURCES src/StandaloneAODProducer.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DataFormatsTPC
        O2::DataFormatsITSMFT
        O2::DataFormatsITS
        O2::DataFormatsFT0
        O2::DataFormatsFDD
        O2::DataFormatsFV0
        O2::DataFormatsTOF
        O2::ITSReconstruction
        O2::FT0Reconstruction
        O2::TPCFastTransformation
        O2::TPCBase
        O2::TPCReconstruction
        O2::TOFBase
        O2::TOFCalibration
        O2::SimConfig
        O2::DataFormatsFT0
        O2::Steer
        O2::ZDCBase
)
