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

o2_add_library(HMPIDWorkflow
               SOURCES src/DataDecoderSpec.cxx
                       src/DataDecoderSpec2.cxx
                       src/DigitsToRawSpec.cxx
                       src/DigitsReaderSpec.cxx
                       src/ClustersReaderSpec.cxx
                       src/DigitsToClustersSpec.cxx
                       src/DigitsWriterSpec.cxx
                       src/ClustersWriterSpec.cxx
                       src/HMPMatchedWriterSpec.cxx
                       src/DumpDigitsSpec.cxx
                       src/PedestalsCalculationSpec.cxx
                       src/RawToDigitsSpec.cxx
                       src/ReadRawFileSpec.cxx
                       src/WriteRawFileSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/HMPMatchedReaderSpec.cxx
                       src/HMPMatchedWriterSpec.cxx

                       PUBLIC_LINK_LIBRARIES
                                     O2::Framework
                                     O2::CCDB
                                     O2::DPLUtils
                                     O2::DetectorsRaw
                                     O2::HMPIDBase
                                     O2::DataFormatsHMP
                                     O2::DataFormatsDCS
                                     O2::HMPIDSimulation
                                     O2::HMPIDReconstruction)

o2_add_executable(digits-reader-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/digits-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(clusters-reader-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/clusters-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME hhmpid
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(read-raw-file-stream-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/read-raw-file-stream-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(raw-to-pedestals-workflow
                 COMPONENT_NAME hmpid
                 SOURCES src/raw-to-pedestals-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(raw-to-digits-workflow
                 COMPONENT_NAME hmpid
                 SOURCES src/raw-to-digits-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(raw-to-digits-stream-workflow
                 COMPONENT_NAME hmpid
                 SOURCES src/raw-to-digits-stream-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(dump-digits-stream-workflow
                 COMPONENT_NAME hmpid
                 SOURCES src/dump-digits-stream-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(digits-to-raw-workflow
                 COMPONENT_NAME hmpid
                 SOURCES src/digits-to-raw-workflow.cxx
                 PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(digits-to-clusters-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/digits-to-clusters-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(digits-writer-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/digits-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(clusters-writer-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/clusters-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(digits-to-raw-stream-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/digits-to-raw-stream-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

