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

o2_add_executable(
  workflow
  COMPONENT_NAME reco-filtering
  SOURCES src/filtering-workflow.cxx src/FilteringSpec.cxx
  PUBLIC_LINK_LIBRARIES
          O2::DataFormatsGlobalTracking
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

o2_add_executable(
  workflow
  COMPONENT_NAME filtered-reco-tf-reader
  SOURCES src/filtered-tf-reader-workflow.cxx
          src/FilteredTFReaderSpec.cxx
  PUBLIC_LINK_LIBRARIES
          O2::Framework
          O2::DataFormatsGlobalTracking
          O2::SimulationDataFormat
          O2::ReconstructionDataFormats
          O2::DataFormatsGlobalTracking
          O2::DetectorsCommonDataFormats
          O2::DataFormatsITSMFT
          O2::DataFormatsITS
)

o2_add_executable(
  workflow
  COMPONENT_NAME filtered-reco-tf-writer
  SOURCES src/filtered-tf-writer-workflow.cxx
          src/FilteredTFWriterSpec.cxx
  PUBLIC_LINK_LIBRARIES
          O2::Framework
          O2::DPLUtils
          O2::DataFormatsGlobalTracking
)
