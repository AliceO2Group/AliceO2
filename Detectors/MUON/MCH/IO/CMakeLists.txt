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

o2_add_library(MCHIO
        SOURCES
           src/ClusterReaderSpec.cxx
           src/ClusterWriterSpec.cxx
           src/DigitReaderSpec.cxx
           src/DigitWriterSpec.cxx
           src/PreClusterReaderSpec.cxx
           src/PreClusterWriterSpec.cxx
           src/TrackReaderSpec.cxx
           src/TrackDumperSpec.cxx
           src/TrackWriterSpec.cxx
        PUBLIC_LINK_LIBRARIES
           O2::DPLUtils
           O2::DataFormatsMCH
           O2::Framework
           O2::MCHBase
           O2::SimulationDataFormat
           )

o2_add_executable(
        clusters-reader-workflow
        SOURCES src/clusters-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(
        clusters-writer-workflow
        SOURCES src/clusters-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHIO)

o2_add_executable(
        digits-reader-workflow
        SOURCES src/digits-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(
        digits-writer-workflow
        SOURCES src/digits-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(
        preclusters-writer-workflow
        SOURCES src/preclusters-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHIO)

o2_add_executable(
        preclusters-reader-workflow
        SOURCES src/preclusters-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(tracks-dumper-workflow
        SOURCES src/tracks-dumper-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(tracks-reader-workflow
        SOURCES src/tracks-reader-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES
          O2::DetectorsRaw
          O2::MCHIO
          )

o2_add_executable(tracks-writer-workflow
        SOURCES src/tracks-writer-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHIO)

