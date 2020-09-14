# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MIDWorkflow
               TARGETVARNAME targetName
               SOURCES src/RecoWorkflow.cxx
                       src/ClusterizerMCSpec.cxx
                       src/DigitReaderSpec.cxx
                       src/RecoWorkflowMC.cxx
                       src/TrackerMCSpec.cxx
		       src/ClusterizerSpec.cxx
		       src/RecoWorkflow.cxx
		       src/RawDecoderSpec.cxx
		       src/RawAggregatorSpec.cxx
		       src/TrackerSpec.cxx
		       src/DigitReaderSpec.cxx
		       src/DigitsToRawWorkflow.cxx
		       src/RawWriterSpec.cxx
		       PUBLIC_LINK_LIBRARIES
		       O2::Framework
		       O2::SimConfig
		       ms_gsl::ms_gsl
		       O2::DetectorsBase
		       O2::SimulationDataFormat
		       O2::DataFormatsMID
		       O2::DPLUtils
		       O2::MIDSimulation
		       O2::MIDClustering
		       O2::MIDTracking
		       O2::MIDRaw
		       )
		     
o2_add_executable(
  reco-workflow-mc
  COMPONENT_NAME mid
  SOURCES src/mid-reco-workflow-mc.cxx
  TARGETVARNAME exenamerecomc
  PUBLIC_LINK_LIBRARIES
    O2::MIDWorkflow)

target_include_directories(
  ${exenamerecomc} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  reco-workflow
  COMPONENT_NAME mid
  SOURCES src/mid-reco-workflow.cxx
  TARGETVARNAME exenamereco
  PUBLIC_LINK_LIBRARIES
    O2::MIDWorkflow)

target_include_directories(
  ${exenamereco} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

o2_add_executable(
  digits-to-raw-workflow
  COMPONENT_NAME mid
  SOURCES src/mid-digits-to-raw.cxx
  TARGETVARNAME exenameraw
  PUBLIC_LINK_LIBRARIES
    O2::MIDWorkflow)

target_include_directories(
  ${exenameraw} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
