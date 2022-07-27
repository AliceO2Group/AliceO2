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

o2_add_library(
  DataFormatsGlobalTracking
  SOURCES src/RecoContainer.cxx
          src/FilteredRecoTF.cxx
  PUBLIC_LINK_LIBRARIES
    O2::DataFormatsTPC
    O2::DataFormatsITSMFT
    O2::DataFormatsITS
    O2::DataFormatsMFT
    O2::DataFormatsMCH
    O2::DataFormatsMID
    O2::DataFormatsFT0
    O2::DataFormatsFV0
    O2::DataFormatsFDD
    O2::DataFormatsZDC
    O2::DataFormatsTOF
    O2::DataFormatsHMP
    O2::DataFormatsEMCAL
    O2::ReconstructionDataFormats
    O2::DataFormatsTRD
    O2::DataFormatsCTP
    O2::DataFormatsCPV
    O2::DataFormatsPHOS
    O2::DataFormatsEMCAL
    O2::GPUDataTypeHeaders
  PRIVATE_LINK_LIBRARIES
    O2::Framework)

o2_target_root_dictionary(
  DataFormatsGlobalTracking
  HEADERS include/DataFormatsGlobalTracking/FilteredRecoTF.h
)
