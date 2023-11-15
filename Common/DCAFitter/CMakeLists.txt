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

# add_compile_options(-O0 -g -fPIC)

o2_add_library(DCAFitter
               TARGETVARNAME targetName
               SOURCES src/DCAFitterN.cxx
                       src/FwdDCAFitterN.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Core
                                     O2::CommonUtils
                                     O2::ReconstructionDataFormats
                                     O2::DataFormatsParameters
                                     O2::DetectorsBase)

o2_target_root_dictionary(DCAFitter
                          HEADERS include/DCAFitter/HelixHelper.h
                                  include/DCAFitter/DCAFitterN.h
                                  include/DCAFitter/FwdDCAFitterN.h)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()


o2_add_test(
  DCAFitterN
  SOURCES test/testDCAFitterN.cxx
  COMPONENT_NAME DCAFitter
  PUBLIC_LINK_LIBRARIES O2::DCAFitter ROOT::Core ROOT::Physics
  LABELS vertexing
  ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
  VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})
