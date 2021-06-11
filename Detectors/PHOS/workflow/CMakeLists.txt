# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(PHOSWorkflow
               SOURCES src/RecoWorkflow.cxx
                       src/ReaderSpec.cxx
                       src/CellConverterSpec.cxx
                       src/RawToCellConverterSpec.cxx
                       src/ClusterizerSpec.cxx
                       src/DigitsPrinterSpec.cxx
                       src/RawWriterSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/WriterSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework O2::DataFormatsPHOS
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
