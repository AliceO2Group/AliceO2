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

o2_add_library(DetectorsRaw
        SOURCES src/RawFileReader.cxx
        src/RawFileWriter.cxx
        src/RawHeaderStream.cxx
        src/HBFUtils.cxx
        src/RDHUtils.cxx
        src/HBFUtilsInitializer.cxx
        src/DistSTFSenderSpec.cxx
        src/RawDumpSpec.cxx
        PUBLIC_LINK_LIBRARIES AliceO2::Common
        FairRoot::Base
        O2::Headers
        O2::CommonDataFormat
        O2::DetectorsCommonDataFormats
        O2::Framework
        O2::DPLUtils
        O2::Algorithm
        FairMQ::FairMQ)

add_subdirectory(TFReaderDD)

o2_target_root_dictionary(DetectorsRaw
        HEADERS include/DetectorsRaw/RawFileReader.h
        include/DetectorsRaw/RawFileWriter.h
        include/DetectorsRaw/HBFUtils.h
        include/DetectorsRaw/RDHUtils.h)

o2_add_executable(file-check
        COMPONENT_NAME raw
        SOURCES src/rawfileCheck.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
        Boost::program_options)

o2_add_executable(file-split
        COMPONENT_NAME raw
        SOURCES src/rawfileSplit.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw)

o2_add_executable(file-reader-workflow
        COMPONENT_NAME raw
        SOURCES src/rawfile-reader-workflow.cxx
        src/RawFileReaderWorkflow.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw)

o2_add_executable(sender-workflow
        COMPONENT_NAME diststf
        SOURCES src/diststf-sender-workflow.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw)

o2_add_executable(data-dump-workflow
        COMPONENT_NAME raw
        SOURCES src/raw-data-dump-workflow.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw)

o2_add_test(HBFUtils
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
        O2::SimulationDataFormat
        SOURCES test/testHBFUtils.cxx
        COMPONENT_NAME raw
        LABELS raw)

o2_add_test(RawReaderWriter
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
        O2::SimulationDataFormat
        O2::DPLUtils
        SOURCES test/testRawReaderWriter.cxx
        COMPONENT_NAME raw
        LABELS raw)

o2_add_test_root_macro(macro/rawStat.C
        PUBLIC_LINK_LIBRARIES O2::DetectorsRaw
        O2::CommonUtils
        LABELS raw)
