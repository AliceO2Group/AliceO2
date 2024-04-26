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

o2_add_library(MCHAlign
  SOURCES
    src/Aligner.cxx
    src/AlignmentSpec.cxx
    src/AlignRecordSpec.cxx
  PUBLIC_LINK_LIBRARIES
    O2::MathUtils
    O2::CCDB
    O2::DataFormatsMCH
    O2::ForwardAlign
    O2::MCHTracking
    O2::GlobalTracking
    O2::GlobalTrackingWorkflow
    O2::MCHGeometryTransformer
    O2::CommonUtils
    O2::DataFormatsParameters
    O2::DetectorsBase
    O2::DetectorsRaw
    O2::Framework
    O2::DetectorsRaw
    O2::Headers
    O2::ReconstructionDataFormats
    O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(MCHAlign
  HEADERS
    include/MCHAlign/Aligner.h
    include/MCHAlign/AlignmentSpec.h
    include/MCHAlign/AlignRecordSpec.h)

o2_add_executable(
        alignment-workflow
        SOURCES src/alignment-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::ForwardAlign
          O2::MCHAlign
          Boost::program_options)

o2_add_executable(
        align-record-workflow
        SOURCES src/align-record-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::MCHAlign)



