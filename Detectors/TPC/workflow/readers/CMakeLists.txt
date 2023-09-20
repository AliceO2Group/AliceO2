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

o2_add_library(TPCReaderWorkflow
               SOURCES src/ClusterReaderSpec.cxx
                       src/PublisherSpec.cxx
                       src/TrackReaderSpec.cxx
                       src/TriggerReaderSpec.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsTPC
                                     O2::DataFormatsGlobalTracking
                                     O2::DPLUtils
                                     O2::TPCBase
           )

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${mergertargetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${mergertargetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
