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

o2_add_library(PHOSWorkflow
               SOURCES src/CellReaderSpec.cxx
                       src/DigitReaderSpec.cxx
                       src/RecoWorkflow.cxx
                       src/ReaderSpec.cxx
                       src/CellConverterSpec.cxx
                       src/RawToCellConverterSpec.cxx
                       src/ClusterizerSpec.cxx
                       src/DigitsPrinterSpec.cxx
                       src/RawWriterSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/WriterSpec.cxx
                       src/EventBuilderSpec.cxx
                       src/StandaloneAODProducerSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsPHOS
                                     O2::MathUtils
                                     O2::DPLUtils
                                     O2::PHOSBase
                                     O2::PHOSSimulation
                                     O2::PHOSReconstruction
                                     O2::Algorithm)

o2_add_executable(reco-workflow
                  COMPONENT_NAME phos
                  SOURCES src/phos-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::PHOSWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME phos
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::PHOSWorkflow)

o2_add_executable(cell-writer-workflow
                  COMPONENT_NAME phos
                  SOURCES src/cell-writer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::PHOSWorkflow)

o2_add_executable(event-builder
                  COMPONENT_NAME phos
                  SOURCES src/event-builder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::PHOSWorkflow)

o2_add_executable(standalone-aod-producer-workflow
                  COMPONENT_NAME phos
                  SOURCES src/phos-standalone-aod-producer-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::PHOSWorkflow O2::MathUtils)
