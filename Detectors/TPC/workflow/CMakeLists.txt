# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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
                       src/FileReaderWorkflow.cxx
                       src/CalDetMergerPublisherSpec.cxx
                       src/KryptonClustererSpec.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsTPC
                                     O2::DPLUtils O2::TPCReconstruction
                                     O2::TPCCalibration O2::TPCSimulation
                                     O2::TPCQC O2::DetectorsCalibration
                                     O2::TPCReaderWorkflow
               PRIVATE_LINK_LIBRARIES O2::GPUTracking # For the Zero Suppression includes
                                      O2::GPUWorkflow
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

o2_add_executable(calib-pedestal
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-calib-pedestal.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(integrate-idc
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-integrate-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(averagegroup-idc
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-averagegroup-idc.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(aggregate-idc
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-aggregate-grouped-idc.cxx
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

o2_add_test(workflow
            COMPONENT_NAME tpc
            LABELS tpc workflow
            SOURCES test/test_TPCWorkflow.cxx
            PUBLIC_LINK_LIBRARIES O2::TPCWorkflow
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_executable(digits-to-rawzs
                  COMPONENT_NAME tpc
                  PUBLIC_LINK_LIBRARIES O2::TPCBase O2::SimulationDataFormat O2::GPUO2Interface O2::GPUTracking O2::DetectorsRaw
                  SOURCES src/convertDigitsToRawZS.cxx)

add_subdirectory(readers)
