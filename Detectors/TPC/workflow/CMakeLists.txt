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

o2_add_library(TPCWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/ClustererSpec.cxx
                       src/ClusterDecoderRawSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/RawToDigitsSpec.cxx
                       src/LinkZSToDigitsSpec.cxx
                       src/ZSSpec.cxx
                       src/CalibProcessingHelper.cxx
                       src/ClusterSharingMapSpec.cxx
                       src/CalDetMergerPublisherSpec.cxx
                       src/KryptonClustererSpec.cxx
                       src/KryptonRawFilterSpec.cxx
                       src/SACProcessorSpec.cxx
                       src/IDCToVectorSpec.cxx
                       src/CalibdEdxSpec.cxx
                       src/CalibratordEdxSpec.cxx
                       src/MIPTrackFilterSpec.cxx
                       src/LaserTrackFilterSpec.cxx
                       src/ApplyCCDBCalibSpec.cxx
                       src/ProcessingHelpers.cxx
                       src/TrackAndClusterFilterSpec.cxx
                       src/FileWriterSpec.cxx
                       src/TPCVDriftTglCalibSpec.cxx
                       src/ClusterQCSpec.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsTPC
                                     O2::DPLUtils O2::TPCReconstruction
                                     O2::TPCCalibration O2::TPCSimulation
                                     O2::TPCQC O2::DetectorsCalibration
                                     O2::TPCReaderWorkflow
               PRIVATE_LINK_LIBRARIES O2::GPUTracking # For the Zero Suppression includes
                                      O2::GPUWorkflow
           )

o2_add_library(TPCWorkflowGUI
               SOURCES src/MonitorWorkflowSpec.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::TPCWorkflow
                                     O2::TPCMonitor
           )


o2_add_executable(chunkeddigit-merger
        COMPONENT_NAME tpc
        TARGETVARNAME mergertargetName
        SOURCES src/ChunkedDigitPublisher.cxx
        PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${mergertargetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${mergertargetName} PRIVATE OpenMP::OpenMP_CXX)
endif()


o2_add_executable(reco-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(raw-to-digits-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-raw-to-digits-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calib-pad-raw
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calib-pad-raw.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(laser-tracks-calibrator
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-laser-tracks-calibrator.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calib-laser-tracks
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calib-laser-tracks.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(laser-track-filter
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-laser-track-filter.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(apply-ccdb-calib
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-apply-ccdb-calib.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(sac-processing
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-sac-processing.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-integrate
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-integrate-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-flp
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-flp-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-distribute
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-distribute-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(sac-distribute
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-distribute-sac.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-ft-aggregator
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-fouriertransform-aggregator.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-ft-epn
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-fouriertransform-epn.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-factorize
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-factorize-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(sac-factorize
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-factorize-sac.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(track-reader
                  COMPONENT_NAME tpc
                  SOURCES src/TrackReaderWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(file-reader
                  COMPONENT_NAME tpc
                  SOURCES src/FileReaderWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(krypton-clusterer
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-krypton-clusterer.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(krypton-raw-filter
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-krypton-raw-filter.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-to-vector
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-idc-to-vector.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calib-dedx
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calib-dEdx.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calibrator-dedx
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calibrator-dEdx.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(idc-test-ft
                COMPONENT_NAME tpc
                SOURCES test/test_ft_EPN_Aggregator.cxx
                PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(miptrack-filter
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-miptrack-filter.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(track-and-cluster-filter
                  COMPONENT_NAME tpc
                  SOURCES src/track-and-cluster-filter.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calib-gainmap-tracks
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calib-gainmap-tracks.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow O2::GlobalTracking)

o2_add_executable(integrate-clusters
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-integrate-cluster-currents.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_test(workflow
            COMPONENT_NAME tpc
            LABELS tpc workflow
            SOURCES test/test_TPCWorkflow.cxx
            PUBLIC_LINK_LIBRARIES O2::TPCWorkflow
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_executable(digits-to-rawzs
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCBase O2::SimulationDataFormat O2::GPUO2Interface O2::GPUTracking O2::DetectorsRaw O2::TPCReconstruction
                  SOURCES src/convertDigitsToRawZS.cxx)

o2_add_executable(monitor-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-monitor-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflowGUI)

o2_add_executable(calibrator-gainmap-tracks
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calibrator-gainmap-tracks.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(cluster-qc
                  SOURCES src/tpc-cluster-qc.cxx
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(calibration-workflow
                  COMPONENT_NAME tpc-vdrift-tgl
                  SOURCES src/tpc-vdrift-tgl-calibration-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::TPCWorkflow
                                        O2::DetectorsCalibration
                                        O2::DetectorsCalibrationWorkflow
                                        O2::DataFormatsCalibration
                                        O2::CCDB)

add_subdirectory(readers)
