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

add_subdirectory(io)

o2_add_library(TRDWorkflow
               TARGETVARNAME targetName
               SOURCES src/TRDDigitizerSpec.cxx
                       src/TRDTrapSimulatorSpec.cxx
                       src/TRDTrackletTransformerSpec.cxx
                       src/TRDEventDisplayFeedSpec.cxx
                       src/TRDGlobalTrackingSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/TrackBasedCalibSpec.cxx
                       src/KrClustererSpec.cxx
                       include/TRDWorkflow/VdAndExBCalibSpec.h
                       include/TRDWorkflow/TRDGlobalTrackingQCSpec.h
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::Steer O2::Algorithm O2::DataFormatsTRD O2::TRDSimulation O2::TRDReconstruction O2::TRDQC O2::DetectorsBase O2::SimulationDataFormat O2::TRDBase O2::TRDCalibration O2::GPUTracking O2::GlobalTrackingWorkflowHelpers O2::GlobalTrackingWorkflowReaders O2::GPUWorkflowHelper O2::ReconstructionDataFormats O2::ITSWorkflow O2::TPCWorkflow O2::TRDWorkflowIO)

o2_add_executable(trap-sim
                  COMPONENT_NAME trd
                  SOURCES src/TRDTrapSimulatorWorkFlow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::DPLUtils
                                        O2::Steer
                                        O2::TRDBase
                                        O2::DataFormatsTRD
                                        O2::TRDWorkflow)

o2_add_executable(global-tracking
                  COMPONENT_NAME trd
                  SOURCES src/trd-tracking-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(tracklet-transformer
                  COMPONENT_NAME trd
                  SOURCES src/TRDTrackletTransformerWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(event-display-feed
                  COMPONENT_NAME trd
                  SOURCES src/TRDEventDisplayFeedWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(trd-vdrift-exb
                  COMPONENT_NAME calibration
                  SOURCES src/trd-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::TRDCalibration O2::TRDWorkflow O2::DetectorsCalibration)

o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME trd
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

o2_add_executable(kr-clusterer
                  COMPONENT_NAME trd
                  SOURCES src/trd-kr-clusterer.cxx
                  PUBLIC_LINK_LIBRARIES O2::TRDWorkflow)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
