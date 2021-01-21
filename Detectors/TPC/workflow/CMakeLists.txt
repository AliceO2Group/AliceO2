# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
                     

# FIXME: do we really need a library here ? Is the exe not enough ?
o2_add_library(TPCWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/PublisherSpec.cxx
                       src/ClustererSpec.cxx
                       src/ClusterDecoderRawSpec.cxx
                       src/CATrackerSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/TrackReaderSpec.cxx
                       src/RawToDigitsSpec.cxx
                       src/LinkZSToDigitsSpec.cxx
                       src/TPCSectorCompletionPolicy.cxx
                       src/ZSSpec.cxx
                       src/CalibProcessingHelper.cxx
                       src/ClusterSharingMapSpec.cxx
                       src/FileReaderWorkflow.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsTPC
                                     O2::DPLUtils O2::TPCReconstruction
                                     O2::TPCCalibration O2::TPCSimulation
                                     O2::DetectorsCalibration)

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

o2_add_executable(track-reader
                  COMPONENT_NAME tpc
                  SOURCES src/TrackReaderWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_executable(file-reader
                  COMPONENT_NAME tpc
                  SOURCES src/FileReaderWorkflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCWorkflow)

o2_add_test(workflow
            COMPONENT_NAME tpc
            LABELS tpc workflow
            SOURCES test/test_TPCWorkflow.cxx
            PUBLIC_LINK_LIBRARIES O2::TPCWorkflow
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

if(GPUCA_EVENT_DISPLAY OR
                       (OPENGL_FOUND AND GLFW_FOUND AND TARGET AliceO2::DebugGUI AND OPENGL_GLU_FOUND
                       AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY)
endif()
