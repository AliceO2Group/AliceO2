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
  FT0Workflow
  SOURCES src/RecoWorkflow.cxx
          src/ReconstructionSpec.cxx
          src/RecPointWriterSpec.cxx
          src/RecPointReaderSpec.cxx
          src/EntropyEncoderSpec.cxx
          src/EntropyDecoderSpec.cxx
          src/DigitReaderSpec.cxx
          src/FT0Workflow.cxx
          src/FT0DataProcessDPLSpec.cxx
          src/FT0DataReaderDPLSpec.cxx
          src/FT0DigitWriterSpec.cxx
          src/RawReaderFT0.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::DataFormatsFT0
    O2::FT0Reconstruction
    O2::DetectorsCommonDataFormats
    O2::DPLUtils
    O2::FT0Raw)

o2_add_executable(
  reco-workflow
  SOURCES src/ft0-reco-workflow.cxx
  COMPONENT_NAME ft0
  PUBLIC_LINK_LIBRARIES O2::FT0Workflow
  TARGETVARNAME fitrecoexe)

o2_add_executable(
  entropy-encoder-workflow
  SOURCES src/entropy-encoder-workflow.cxx
  COMPONENT_NAME ft0
  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)

o2_add_executable(
  digits-reader-workflow
  SOURCES src/digits-reader-workflow.cxx
  COMPONENT_NAME ft0
  PUBLIC_LINK_LIBRARIES O2::FT0Workflow)

o2_add_executable(
  flp-dpl-workflow
  COMPONENT_NAME ft0
  SOURCES src/ft0-flp-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::FT0Workflow
  TARGETVARNAME ft0flpexe)

if(NOT
   APPLE)

  set_property(
    TARGET ${fitrecoexe}
    PROPERTY LINK_WHAT_YOU_USE
             ON)

endif()
