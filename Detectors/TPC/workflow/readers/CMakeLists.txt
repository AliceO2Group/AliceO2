# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
                     
o2_add_library(TPCReaderWorkflow
               SOURCES src/ClusterReaderSpec.cxx
                       src/PublisherSpec.cxx
                       src/TrackReaderSpec.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DataFormatsTPC
                                     O2::DPLUtils
                                     O2::TPCBase
           )

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${mergertargetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${mergertargetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
