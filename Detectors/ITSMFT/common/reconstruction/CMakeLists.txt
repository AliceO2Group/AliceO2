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

o2_add_library(ITSMFTReconstruction
               TARGETVARNAME targetName
               SOURCES src/ChipMappingITS.cxx
                       src/ChipMappingMFT.cxx
                       src/DigitPixelReader.cxx
                       src/Clusterer.cxx
                       src/ClustererParam.cxx
                       src/PixelData.cxx
                       src/BuildTopologyDictionary.cxx
                       src/LookUp.cxx
                       src/TopologyFastSimulation.cxx
                       src/AlpideCoder.cxx
                       src/PayLoadCont.cxx
                       src/GBTWord.cxx
                       src/RUInfo.cxx
                       src/GBTLink.cxx
                       src/RUDecodeData.cxx
                       src/RawPixelDecoder.cxx
                       src/CTFCoder.cxx
                       src/DecodingStat.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                                     O2::CommonDataFormat
                                     O2::DetectorsRaw
                                     O2::DetectorsBase
                                     O2::SimulationDataFormat 
                                     O2::DataFormatsITSMFT
                                     O2::DPLUtils
                                     O2::rANS
                                     O2::Headers)

o2_target_root_dictionary(
  ITSMFTReconstruction
  HEADERS include/ITSMFTReconstruction/PixelReader.h
          include/ITSMFTReconstruction/GBTLink.h
          include/ITSMFTReconstruction/RUDecodeData.h
          include/ITSMFTReconstruction/DigitPixelReader.h
          include/ITSMFTReconstruction/RawPixelReader.h
          include/ITSMFTReconstruction/PixelData.h
          include/ITSMFTReconstruction/Clusterer.h
          include/ITSMFTReconstruction/ClustererParam.h
          include/ITSMFTReconstruction/BuildTopologyDictionary.h
          include/ITSMFTReconstruction/LookUp.h
          include/ITSMFTReconstruction/TopologyFastSimulation.h
          include/ITSMFTReconstruction/ChipMappingITS.h
          include/ITSMFTReconstruction/ChipMappingMFT.h
          include/ITSMFTReconstruction/AlpideCoder.h
          include/ITSMFTReconstruction/GBTWord.h
          include/ITSMFTReconstruction/PayLoadCont.h
          include/ITSMFTReconstruction/PayLoadSG.h
          include/ITSMFTReconstruction/DecodingStat.h
          include/ITSMFTReconstruction/RUInfo.h)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
  
