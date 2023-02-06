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

o2_add_library(FV0Workflow
               SOURCES src/RecoWorkflow.cxx
                       src/ReconstructionSpec.cxx
                       src/RecPointWriterSpec.cxx
                       src/RecPointReaderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/DigitReaderSpec.cxx
                       src/FV0DigitWriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsFV0
                                     O2::FV0Reconstruction
                                     O2::FV0Raw
                                     O2::DetectorsCommonDataFormats
                                     O2::Framework
                                     O2::DPLUtils
                                     O2::CommonUtils)

o2_add_executable(reco-workflow
                  SOURCES src/fv0-reco-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow
                  TARGETVARNAME fv0recoexe)

o2_add_executable(entropy-encoder-workflow
                  SOURCES src/entropy-encoder-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow)

o2_add_executable(digit-reader-workflow
                  SOURCES src/digits-reader-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow)

o2_add_executable(digits-writer-workflow
                  SOURCES src/digits-writer-workflow.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow)

o2_add_executable(flp-dpl-workflow
                  COMPONENT_NAME fv0
                  SOURCES src/fv0-flp-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow O2::FITWorkflow O2::FV0Raw
                  TARGETVARNAME fv0flpexe)

o2_add_executable(integrate-cluster-workflow
                  SOURCES src/cluster-integrator.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow O2::FITWorkflow)

o2_add_executable(integrate-cluster-reader-workflow
                  SOURCES src/cluster-integrator-reader.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow O2::FITWorkflow)

o2_add_executable(merge-integrate-cluster-workflow
                  SOURCES src/cluster-merge-integrator.cxx
                  COMPONENT_NAME fv0
                  PUBLIC_LINK_LIBRARIES O2::FV0Workflow O2::FITWorkflow)

if(NOT APPLE)

 set_property(TARGET ${fitrecoexe} PROPERTY LINK_WHAT_YOU_USE ON)

endif()

