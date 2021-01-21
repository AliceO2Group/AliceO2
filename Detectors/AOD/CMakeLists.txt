# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  AODProducerWorkflow
  SOURCES src/AODProducerWorkflow.cxx
          src/AODProducerWorkflowSpec.cxx
  PUBLIC_LINK_LIBRARIES
    O2::AnalysisDataModel
    O2::DetectorsVertexing
    O2::Framework
    O2::FT0Workflow
    O2::GlobalTracking
    O2::GlobalTrackingWorkflow
    O2::ITSMFTWorkflow
    O2::ITStracking
    O2::ITSWorkflow
    O2::SimulationDataFormat
    O2::Steer
    O2::TOFWorkflow
    O2::TPCWorkflow
    O2::CCDB
)

o2_add_executable(
  workflow
  COMPONENT_NAME aod-producer
  SOURCES src/aod-producer-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::AODProducerWorkflow
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
    O2::AnalysisDataModel
    O2::Steer
)



