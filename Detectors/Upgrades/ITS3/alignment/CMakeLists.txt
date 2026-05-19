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

# add_compile_options(-O0 -g -fPIC -fno-omit-frame-pointer)
o2_add_library(ITS3Align
               TARGETVARNAME targetName
               SOURCES src/AlignmentHierarchy.cxx
                       src/AlignmentDOF.cxx
                       src/AlignmentMath.cxx
                       src/MisalignmentUtils.cxx
                       src/AlignmentSensors.cxx
                       src/AlignmentParams.cxx
                       src/AlignmentTypes.cxx
                       src/AlignmentSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::MathUtils
                                     O2::Steer
                                     O2::ITSBase
                                     O2::ITSMFTSimulation
                                     O2::ITS3Reconstruction
                                     O2::Framework
                                     O2::GlobalTrackingWorkflowReaders
                                     O2::GlobalTrackingWorkflowHelpers
                                     O2::DataFormatsGlobalTracking
                                     O2::DetectorsVertexing
                                     nlohmann_json::nlohmann_json
                                     GBL::GBL)
if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_target_root_dictionary(ITS3Align
                          HEADERS include/ITS3Align/AlignmentParams.h
                                  include/ITS3Align/AlignmentTypes.h)


o2_add_executable(alignment-workflow
                  SOURCES src/alignment-workflow.cxx
                  COMPONENT_NAME its3
                  PUBLIC_LINK_LIBRARIES O2::ITS3Align)
