# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EndCapsReconstruction
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
               PUBLIC_LINK_LIBRARIES O2::EndCapsBase
                                     O2::CommonDataFormat
	                             O2::DetectorsRaw
                                     O2::SimulationDataFormat
                                     O2::DataFormatsITSMFT
				     O2::DPLUtils
                                     O2::rANS
                                     O2::Headers)

o2_target_root_dictionary(
  EndCapsReconstruction
  HEADERS include/EndCapsReconstruction/PixelReader.h
          include/EndCapsReconstruction/GBTLink.h
          include/EndCapsReconstruction/RUDecodeData.h
	  include/EndCapsReconstruction/RawPixelDecoder.h
          include/EndCapsReconstruction/DigitPixelReader.h
          include/EndCapsReconstruction/RawPixelReader.h
          include/EndCapsReconstruction/PixelData.h
          include/EndCapsReconstruction/Clusterer.h
          include/EndCapsReconstruction/ClustererParam.h
          include/EndCapsReconstruction/BuildTopologyDictionary.h
          include/EndCapsReconstruction/LookUp.h
          include/EndCapsReconstruction/TopologyFastSimulation.h
          include/EndCapsReconstruction/ChipMappingITS.h
          include/EndCapsReconstruction/ChipMappingMFT.h
          include/EndCapsReconstruction/AlpideCoder.h
          include/EndCapsReconstruction/GBTWord.h
          include/EndCapsReconstruction/PayLoadCont.h
          include/EndCapsReconstruction/PayLoadSG.h
          include/EndCapsReconstruction/RUInfo.h)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
