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

o2_add_library(DetectorsCommonDataFormats
               SOURCES src/DetID.cxx src/AlignParam.cxx src/DetMatrixCache.cxx
                       src/DetectorNameConf.cxx
                       src/EncodedBlocks.cxx
                       src/CTFHeader.cxx
                       src/CTFDictHeader.cxx
                       src/CTFIOSize.cxx
         src/FileMetaData.cxx
               PUBLIC_LINK_LIBRARIES
               ROOT::Core
               ROOT::Geom
               O2::GPUCommon
               O2::GPUUtils
               O2::MathUtils
               O2::FrameworkLogger
               O2::Headers
               O2::rANS
               O2::Framework
               O2::CommonUtils)

o2_target_root_dictionary(
  DetectorsCommonDataFormats
  HEADERS include/DetectorsCommonDataFormats/DetID.h
          include/DetectorsCommonDataFormats/SimTraits.h
          include/DetectorsCommonDataFormats/AlignParam.h
          include/DetectorsCommonDataFormats/DetectorNameConf.h
          include/DetectorsCommonDataFormats/EncodedBlocks.h
          include/DetectorsCommonDataFormats/CTFHeader.h
          include/DetectorsCommonDataFormats/CTFDictHeader.h
          include/DetectorsCommonDataFormats/CTFIOSize.h
          include/DetectorsCommonDataFormats/DetMatrixCache.h)

configure_file(UpgradesStatus.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/DetectorsCommonDataFormats/UpgradesStatus.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/DetectorsCommonDataFormats/UpgradesStatus.h
        DESTINATION include/DetectorsCommonDataFormats)

o2_add_test(DetID
            SOURCES test/testDetID.cxx
            PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
            COMPONENT_NAME DetectorsCommonDataFormats
            LABELS dataformats)
