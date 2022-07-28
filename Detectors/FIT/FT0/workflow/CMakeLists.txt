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

CHECK_CXX_COMPILER_FLAG("-mavx512f -mavx512vl -mavx512bw -mavx512dq" FT0_DECODER_AVX512_GOOD_FLAGS)
if((DEFINED ENV{O2_BUILD_FOR_FLP}) AND (CMAKE_SYSTEM_NAME STREQUAL "Linux") AND FT0_DECODER_AVX512_GOOD_FLAGS)
add_definitions(-DFT0_DECODER_AVX512)
o2_add_library(FT0Workflow
               SOURCES src/RecoWorkflow.cxx
                       src/ReconstructionSpec.cxx
                       src/RecPointWriterSpec.cxx
                       src/RecPointReaderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/DigitReaderSpec.cxx
                       src/FT0DigitWriterSpec.cxx
                       src/RecoQCworkflow.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsFT0
                                     O2::FT0Reconstruction
                                     O2::FT0Raw
                                     O2::FT0Decoder
                                     O2::DetectorsCommonDataFormats
                                     O2::Framework
                                     O2::DPLUtils
                                     O2::DataFormatsGlobalTracking)
o2_add_library(FT0Decoder
               SOURCES src/FT0DataDecoderDPLSpec.cxx
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsFT0
                                     O2::Framework
                                     O2::DetectorsCommonDataFormats
                                     O2::DPLUtils
                                     TARGETVARNAME targetDecoderAVX512)
target_compile_options(${targetDecoderAVX512} PRIVATE -mavx512f -mavx512vl -mavx512bw -mavx512dq)
else()
o2_add_library(FT0Workflow
               SOURCES src/RecoWorkflow.cxx
                       src/ReconstructionSpec.cxx
                       src/RecPointWriterSpec.cxx
                       src/RecPointReaderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/DigitReaderSpec.cxx
                       src/FT0DigitWriterSpec.cxx
                       src/RecoQCworkflow.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsFT0
                                     O2::FT0Reconstruction
                                     O2::FT0Raw
                                     O2::DetectorsCommonDataFormats
                                     O2::Framework
                                     O2::DPLUtils
                                     O2::DataFormatsGlobalTracking)

endif()

o2_add_executable(reco-workflow
                  SOURCES src/ft0-reco-workflow.cxx
                  COMPONENT_NAME ft0
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow
                  TARGETVARNAME fitrecoexe)

o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME ft0
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)

o2_add_executable(digits-reader-workflow
                  SOURCES src/digits-reader-workflow.cxx
                  COMPONENT_NAME ft0
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)

o2_add_executable(digits-writer-workflow
                  SOURCES src/digits-writer-workflow.cxx
                  COMPONENT_NAME ft0
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)

o2_add_executable(flp-dpl-workflow
                  COMPONENT_NAME ft0
                  SOURCES src/ft0-flp-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow  O2::FT0Raw O2::FITWorkflow
                  TARGETVARNAME ft0flpexe)

o2_add_executable(reco-qc
                  COMPONENT_NAME ft0
                  SOURCES src/reco-qc.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::FT0Workflow O2::GlobalTrackingWorkflowHelpers)

o2_add_executable(recpoints-reader-workflow
                  SOURCES src/recpoints-reader-workflow.cxx
                  COMPONENT_NAME ft0
                  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)
if(NOT APPLE)

 set_property(TARGET ${fitrecoexe} PROPERTY LINK_WHAT_YOU_USE ON)

endif()

