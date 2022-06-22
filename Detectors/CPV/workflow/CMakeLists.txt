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

o2_add_library(CPVWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/ReaderSpec.cxx
                       src/WriterSpec.cxx
                       src/ClusterizerSpec.cxx
                       src/DigitsPrinterSpec.cxx
                       src/RawToDigitConverterSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsCPV
                                     O2::DataFormatsCTP
                                     O2::DPLUtils
                                     O2::CPVBase
                                     O2::CPVSimulation
                                     O2::CPVReconstruction
                                     O2::Algorithm)

o2_add_executable(reco-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/cpv-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CPVWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CPVWorkflow)

o2_add_executable(cluster-writer-workflow
                  COMPONENT_NAME cpv
                  SOURCES src/cluster-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::CPVWorkflow)
