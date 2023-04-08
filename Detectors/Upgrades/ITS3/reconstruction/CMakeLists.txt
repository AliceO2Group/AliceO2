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

o2_add_library(ITS3Reconstruction
               TARGETVARNAME targetName
               SOURCES src/Clusterer.cxx
                       src/TopologyDictionary.cxx
                       src/BuildTopologyDictionary.cxx
                       src/LookUp.cxx
                       src/IOUtils.cxx
                #        src/FastMultEst.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                                     O2::ITSMFTReconstruction
                                     O2::ITS3Base
                                     O2::ITStracking
                                     O2::CommonDataFormat
                                     O2::DetectorsRaw
                                     O2::SimulationDataFormat
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsITS3
                                     O2::DPLUtils
                                     O2::rANS
                                     O2::Headers)

o2_target_root_dictionary(
  ITS3Reconstruction
  HEADERS include/ITS3Reconstruction/Clusterer.h
          include/ITS3Reconstruction/TopologyDictionary.h
          include/ITS3Reconstruction/BuildTopologyDictionary.h
          include/ITS3Reconstruction/LookUp.h
          include/ITS3Reconstruction/IOUtils.h
)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
