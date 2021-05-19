# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  ITSMFTWorkflow
  SOURCES src/ClusterReaderSpec.cxx
          src/DigitWriterSpec.cxx
          src/DigitReaderSpec.cxx
          src/STFDecoderSpec.cxx
          src/EntropyEncoderSpec.cxx
          src/EntropyDecoderSpec.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DataFormatsITSMFT
    O2::SimulationDataFormat
    O2::ITSMFTReconstruction
    O2::ITSBase
    O2::MFTBase
    O2::DetectorsCommonDataFormats
    O2::DataFormatsParameters)

o2_add_executable(
  stf-decoder-workflow
  SOURCES src/stf-decoder-workflow.cxx
  COMPONENT_NAME itsmft
  PUBLIC_LINK_LIBRARIES O2::ITSMFTWorkflow)

o2_add_executable(
  entropy-encoder-workflow
  SOURCES src/entropy-encoder-workflow.cxx
  COMPONENT_NAME itsmft
  PUBLIC_LINK_LIBRARIES O2::ITSMFTWorkflow)

o2_add_executable(
  digit-writer-workflow
  SOURCES src/digit-writer-workflow.cxx
  COMPONENT_NAME itsmft
  PUBLIC_LINK_LIBRARIES O2::ITSMFTWorkflow)

o2_add_executable(
  digit-reader-workflow
  SOURCES src/digit-reader-workflow.cxx
  COMPONENT_NAME itsmft
  PUBLIC_LINK_LIBRARIES O2::ITSMFTWorkflow)
