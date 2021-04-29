# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(ITS3Reconstruction
               TARGETVARNAME targetName
               SOURCES src/Clusterer.cxx
                       src/ClustererSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::ITSMFTBase
                                     O2::ITSMFTReconstruction
                                     O2::ITS3Base
                                     O2::CommonDataFormat
                                     O2::DetectorsRaw
                                     O2::SimulationDataFormat 
                                     O2::DataFormatsITSMFT
                                     O2::DPLUtils
                                     O2::rANS
                                     O2::Headers)

o2_target_root_dictionary(
  ITS3Reconstruction
  HEADERS include/ITS3Reconstruction/Clusterer.h
          include/ITS3Reconstruction/ClustererSpec.h

)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
  
