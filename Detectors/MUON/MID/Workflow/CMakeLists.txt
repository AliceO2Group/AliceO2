# Copyright 2019-2022 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

o2_add_library(
  MIDWorkflow
  TARGETVARNAME targetName
  SOURCES src/CalibDataProcessorSpec.cxx
          src/ClusterizerSpec.cxx
          src/ColumnDataSpecsUtils.cxx
          src/DecodedDataAggregatorSpec.cxx
          src/DigitReaderSpec.cxx
          src/EntropyDecoderSpec.cxx
          src/EntropyEncoderSpec.cxx
          src/FilteringSpec.cxx
          src/FilteringBCSpec.cxx
          src/MaskHandlerSpec.cxx
          src/MaskMakerSpec.cxx
          src/ChannelCalibratorSpec.cxx
          src/RawCheckerSpec.cxx
          src/RawDecoderSpec.cxx
          src/RawDumpSpec.cxx
          src/RawGBTDecoderSpec.cxx
          src/RawInputSpecHandler.cxx
          src/RawWriterSpec.cxx
          src/TimingSpec.cxx
          src/TrackerSpec.cxx
          src/TrackReaderSpec.cxx
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
    O2::MIDCalibration
    O2::MIDClustering
    O2::MIDCTF
    O2::MIDFiltering
    O2::MIDQC
    O2::MIDRaw
    O2::MIDSimulation
    O2::MIDTracking)

o2_add_executable(
  digits-to-raw-workflow
  COMPONENT_NAME mid
  SOURCES src/digits-to-raw-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  entropy-encoder-workflow
  COMPONENT_NAME mid
  SOURCES src/entropy-encoder-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  raw-to-digits-workflow
  COMPONENT_NAME mid
  SOURCES src/raw-to-digits-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  reco-workflow
  COMPONENT_NAME mid
  SOURCES src/reco-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  digits-reader-workflow
  COMPONENT_NAME mid
  SOURCES src/digits-reader-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  raw-checker-workflow
  COMPONENT_NAME mid
  SOURCES src/raw-checker-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  tracks-reader-workflow
  COMPONENT_NAME mid
  SOURCES src/tracks-reader-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  mask-maker-workflow
  COMPONENT_NAME mid
  SOURCES src/mask-maker-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  decoded-digits-writer-workflow
  COMPONENT_NAME mid
  SOURCES src/decoded-digits-writer-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  calibration-workflow
  COMPONENT_NAME mid
  SOURCES src/calibration-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)

o2_add_executable(
  raw-dump-workflow
  COMPONENT_NAME mid
  SOURCES src/raw-dump-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::MIDWorkflow)
