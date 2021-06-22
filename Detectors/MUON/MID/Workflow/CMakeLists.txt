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
  MIDWorkflow
  TARGETVARNAME targetName
  SOURCES src/ClusterizerMCSpec.cxx
          src/ClusterizerSpec.cxx
          src/DigitReaderSpec.cxx
          src/EntropyDecoderSpec.cxx
          src/EntropyEncoderSpec.cxx
          src/RawAggregatorSpec.cxx
          src/RawCheckerSpec.cxx
          src/RawDecoderSpec.cxx
          src/RawGBTDecoderSpec.cxx
          src/RawInputSpecHandler.cxx
          src/RawWriterSpec.cxx
          src/TrackerMCSpec.cxx
          src/TrackerSpec.cxx
          src/ZeroSuppressionSpec.cxx
  PUBLIC_LINK_LIBRARIES
    O2::Framework
    O2::SimConfig
    Microsoft.GSL::GSL
    O2::CommonUtils
    O2::DetectorsBase
    O2::DetectorsCommonDataFormats
    O2::DPLUtils
    O2::SimulationDataFormat
    O2::DataFormatsMID
    O2::MIDClustering
    O2::MIDCTF
    O2::MIDQC
    O2::MIDRaw
    O2::MIDSimulation
    O2::MIDTracking)

o2_add_executable(
  digits-to-raw-workflow
  COMPONENT_NAME mid
  SOURCES src/digits-to-raw-workflow.cxx TARGETVARNAME exenameraw
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenameraw} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  entropy-encoder-workflow
  COMPONENT_NAME mid
  SOURCES src/entropy-encoder-workflow.cxx TARGETVARNAME exenameentropy
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenameentropy}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  raw-to-digits-workflow
  COMPONENT_NAME mid
  SOURCES src/raw-to-digits-workflow.cxx TARGETVARNAME exenamedigitstoraw
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenamedigitstoraw}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  reco-workflow
  COMPONENT_NAME mid
  SOURCES src/reco-workflow.cxx TARGETVARNAME exenamereco
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenamereco} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  digits-reader-workflow
  COMPONENT_NAME mid
  SOURCES src/digits-reader-workflow.cxx TARGETVARNAME exenamedigitsreader
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenamedigitsreader}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  raw-checker-workflow
  COMPONENT_NAME mid
  SOURCES src/raw-checker-workflow.cxx TARGETVARNAME exenamerawchecker
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

target_include_directories(
  ${exenamerawchecker}
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  tracks-reader-workflow
  SOURCES src/tracks-reader-workflow.cxx
  COMPONENT_NAME mid
  PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::DataFormatsMID)
