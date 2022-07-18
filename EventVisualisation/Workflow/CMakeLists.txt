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


if(ALIGPU_BUILD_TYPE STREQUAL "O2"
  AND OPENGL_FOUND
  AND GLFW_FOUND
  AND TARGET AliceO2::DebugGUI
  AND OPENGL_GLU_FOUND
)
  o2_add_executable(export-workflow
          COMPONENT_NAME eve
          TARGETVARNAME exportWorkflowTargetName
          SOURCES
                src/O2DPLDisplay.cxx
                src/DetectorData.cxx
                src/FileProducer.cxx
                src/EveWorkflowHelper.cxx
                src/EveConfiguration.cxx
          PUBLIC_LINK_LIBRARIES
                O2::DataFormatsGlobalTracking
                O2::DetectorsRaw
                O2::DetectorsVertexing
                O2::EventVisualisationBase
                O2::EventVisualisationDetectors
                O2::FrameworkFoundation
                O2::FT0Workflow
                O2::GlobalTrackingWorkflowHelpers
                O2::GlobalTrackingWorkflowReaders
                O2::ITSMFTWorkflow
                O2::MFTWorkflow
                O2::TOFBase
                O2::PHOSBase
                O2::EMCALBase
                O2::MIDBase
                O2::TOFWorkflowIO
                O2::TPCReconstruction
                O2::TPCWorkflow
                O2::TRDBase
                O2::TRDWorkflowIO
                O2::SpacePoints
          )
  target_include_directories(${exportWorkflowTargetName} PUBLIC "include")

  o2_add_executable(aodconverter
          COMPONENT_NAME eve
          TARGETVARNAME coverterTargetName
          SOURCES
                src/AO2DConverter.cxx
                src/DetectorData.cxx
                src/FileProducer.cxx
                src/EveWorkflowHelper.cxx
                src/EveConfiguration.cxx
          PUBLIC_LINK_LIBRARIES
                O2::DataFormatsGlobalTracking
                O2::DetectorsRaw
                O2::DetectorsVertexing
                O2::EventVisualisationBase
                O2::EventVisualisationDetectors
                O2::FrameworkFoundation
                O2::FT0Workflow
                O2::GlobalTrackingWorkflowHelpers
                O2::GlobalTrackingWorkflowReaders
                O2::ITSMFTWorkflow
                O2::MFTWorkflow
                O2::TOFBase
                O2::PHOSBase
                O2::EMCALBase
                O2::MIDBase
                O2::TOFWorkflowIO
                O2::TPCReconstruction
                O2::TPCWorkflow
                O2::TRDBase
                O2::TRDWorkflowIO
                O2::SpacePoints
          )
  target_include_directories(${coverterTargetName} PUBLIC "include")
endif()
