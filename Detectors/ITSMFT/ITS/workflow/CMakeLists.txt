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

o2_add_library(ITSWorkflow
               TARGETVARNAME targetName
               SOURCES src/RecoWorkflow.cxx
                       src/ClusterWriterWorkflow.cxx
                       src/ClustererSpec.cxx
                       src/ClusterWriterSpec.cxx
                       src/TrackerSpec.cxx
                       src/CookedTrackerSpec.cxx
                       src/TrackWriterSpec.cxx
                       src/TrackReaderSpec.cxx
                       src/VertexReaderSpec.cxx
                       src/ThresholdCalibratorSpec.cxx
                       src/ThresholdAggregatorWorkflow.cxx
                       src/ThresholdAggregatorSpec.cxx
                       src/DCSParserWorkflow.cxx
                       src/DCSParserSpec.cxx
         PUBLIC_LINK_LIBRARIES O2::Framework
                               O2::SimConfig
                               O2::DataFormatsITS
                               O2::DataFormatsDCS
                               O2::DataFormatsTRD
                               O2::DataFormatsGlobalTracking
                               O2::SimulationDataFormat
                               O2::ITStracking
                               O2::ITSReconstruction
                               O2::ITSMFTReconstruction
                               O2::ITSMFTWorkflow
                               O2::DetectorsCalibration
                               O2::GlobalTrackingWorkflowWriters
                               O2::CCDB
                               O2::GPUTracking
                               O2::GPUWorkflow)

o2_add_executable(reco-workflow
                  SOURCES src/its-reco-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow
                                        O2::GlobalTrackingWorkflowReaders)

o2_add_executable(cluster-writer-workflow
                  SOURCES src/its-cluster-writer-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

o2_add_executable(cluster-reader-workflow
                  SOURCES src/its-cluster-reader-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

o2_add_executable(threshold-calib-workflow
                  SOURCES src/its-threshold-calib-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

o2_add_executable(threshold-aggregator-workflow
                  SOURCES src/its-threshold-aggregator-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

o2_add_executable(dcs-parser-workflow
                  SOURCES src/its-dcs-parser-workflow.cxx
                  COMPONENT_NAME its
                  PUBLIC_LINK_LIBRARIES O2::ITSWorkflow)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
