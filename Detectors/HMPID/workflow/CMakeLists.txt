# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(HMPIDWorkflow
               SOURCES src/DataDecoderSpec.cxx
                       src/DataDecoderSpec2.cxx
                       src/DigitsToRawSpec.cxx
                       src/DumpDigitsSpec.cxx
                       src/PedestalsCalculationSpec.cxx
                       src/RawToDigitsSpec.cxx
                       src/ReadRawFileSpec.cxx
                       src/WriteRawFileSpec.cxx
                       src/EntropyEncoderSpec.cxx
                       src/EntropyDecoderSpec.cxx

               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::CCDB
                                     O2::DPLUtils
                                     O2::DetectorsRaw
                                     O2::HMPIDBase
                                     O2::DataFormatsHMP
                                     O2::HMPIDSimulation
                                     O2::HMPIDReconstruction)
                                     
#o2_add_executable(recoworkflow
#              COMPONENT_NAME hmpid
#              SOURCES src/HMPIDRecoWorkflow.cxx 
#              PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)
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

o2_add_executable(digits-to-raw-stream-workflow
              COMPONENT_NAME hmpid
              SOURCES src/digits-to-raw-stream-workflow.cxx
              PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)

o2_add_executable(entropy-encoder-workflow
                  COMPONENT_NAME hmpid
                  SOURCES src/entropy-encoder-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::HMPIDWorkflow)
