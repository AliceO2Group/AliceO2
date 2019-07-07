# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITSMFTReconstruction
               SOURCES src/ChipMappingITS.cxx
                       src/ChipMappingMFT.cxx
                       src/DigitPixelReader.cxx
                       src/Clusterer.cxx
                       src/PixelData.cxx
                       src/BuildTopologyDictionary.cxx
                       src/LookUp.cxx
                       src/TopologyFastSimulation.cxx
                       src/AlpideCoder.cxx
                       src/PayLoadCont.cxx
                       src/GBTWord.cxx
                       src/RUInfo.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                                     O2::CommonDataFormat
                                     O2::SimulationDataFormat # FIXME: why do we
                                                              # depend on sim
                                                              # data format for
                                                              # reconstruction ?
                                     O2::DataFormatsITSMFT
                                     O2::Headers)

o2_target_root_dictionary(
  ITSMFTReconstruction
  HEADERS include/ITSMFTReconstruction/PixelReader.h
          include/ITSMFTReconstruction/DigitPixelReader.h
          include/ITSMFTReconstruction/RawPixelReader.h
          include/ITSMFTReconstruction/PixelData.h
          include/ITSMFTReconstruction/Clusterer.h
          include/ITSMFTReconstruction/BuildTopologyDictionary.h
          include/ITSMFTReconstruction/LookUp.h
          include/ITSMFTReconstruction/TopologyFastSimulation.h
          include/ITSMFTReconstruction/ChipMappingITS.h
          include/ITSMFTReconstruction/ChipMappingMFT.h
          include/ITSMFTReconstruction/AlpideCoder.h
          include/ITSMFTReconstruction/GBTWord.h
          include/ITSMFTReconstruction/PayLoadCont.h
          include/ITSMFTReconstruction/PayLoadSG.h
          include/ITSMFTReconstruction/RUInfo.h)
