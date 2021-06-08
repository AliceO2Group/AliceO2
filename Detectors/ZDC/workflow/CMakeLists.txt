# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ZDCWorkflow
               SOURCES src/DigitReaderSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx
                       src/ZDCDataReaderDPLSpec.cxx
                       src/ZDCDigitWriterDPLSpec.cxx
                       src/ZDCRecoWriterDPLSpec.cxx
                       src/RecoWorkflow.cxx
                       src/DigitRecoSpec.cxx
                       src/DigitReaderSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsZDC
                                     O2::ZDCRaw
                                     O2::SimulationDataFormat
                                     O2::DPLUtils
                                     O2::ZDCReconstruction
                                     O2::DataFormatsZDC)

o2_add_executable(raw2digits
                  COMPONENT_NAME zdc
                  SOURCES src/o2-zdc-raw2digits.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow)
                  
o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME zdc
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow)

o2_add_executable(digits-read
                  COMPONENT_NAME zdc
                  SOURCES src/digits-reader-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow)

o2_add_executable(digits-reco
                  COMPONENT_NAME zdc
                  SOURCES src/zdc-reco-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow)

o2_add_executable(write-reco
                  COMPONENT_NAME zdc
                  SOURCES src/ZDCRecoWriterDPLSpec.cxx
                  PUBLIC_LINK_LIBRARIES O2::ZDCWorkflow)
