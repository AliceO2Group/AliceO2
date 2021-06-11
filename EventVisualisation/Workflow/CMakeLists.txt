# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.


if(ALIGPU_BUILD_TYPE STREQUAL "O2"
  AND OPENGL_FOUND
  AND GLFW_FOUND
  AND TARGET AliceO2::DebugGUI
  AND OPENGL_GLU_FOUND
  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin"
)
  o2_add_executable(display
          COMPONENT_NAME eve
          TARGETVARNAME targetName
          SOURCES
                                src/O2DPLDisplay.cxx
                                src/FileProducer.cxx
          PUBLIC_LINK_LIBRARIES
                                O2::EventVisualisationBase
                                O2::EventVisualisationDetectors
                                O2::GPUO2Interface
                                O2::GPUWorkflowHelper
                                O2::DataFormatsGlobalTracking
                                O2::TPCFastTransformation
                                O2::TRDBase
                                O2::TOFBase
                                O2::TPCReconstruction
                                O2::GlobalTrackingWorkflowHelpers)
  target_include_directories(${targetName} PUBLIC "include")
endif()
