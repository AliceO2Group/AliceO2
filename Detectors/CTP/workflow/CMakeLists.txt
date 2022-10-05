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

o2_add_library(CTPWorkflow
               SOURCES src/RawDecoderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsCTP
                                     O2::DPLUtils
                                     O2::DetectorsRaw
                                     O2::Algorithm
                                     O2::CTPReconstruction
                                     O2::CTPWorkflowIO)
o2_add_executable(reco-workflow
                  COMPONENT_NAME ctp
                  SOURCES src/ctp-raw-decoder.cxx
                  PUBLIC_LINK_LIBRARIES O2::Algorithm
                                        O2::CTPWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME ctp
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CTPWorkflow)
