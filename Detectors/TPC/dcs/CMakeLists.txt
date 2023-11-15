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

o2_add_library(TPCdcs
               TARGETVARNAME targetName
               SOURCES src/DCSProcessor.cxx
                       src/DCSConfigSpec.cxx
                       src/DCSSpec.cxx
                       src/DCSDPHints.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::DetectorsDCS
                                     O2::DataFormatsTPC
                                     O2::TPCBase
                                     ROOT::Minuit)

o2_target_root_dictionary(TPCdcs
                          HEADERS include/TPCdcs/DCSProcessor.h)

o2_add_executable(dcs-sim-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-dcs-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCdcs
                                        O2::DCStestWorkflow)
o2_add_executable(dcs-data-replay
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-dcs-data-replay.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCdcs
                                        O2::DCStestWorkflow)

o2_add_executable(dcs-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-dcs-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCdcs)

o2_add_executable(dcs-config-workflow
                  COMPONENT_NAME tpc
                  SOURCES src/tpc-dcs-config-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCdcs)

o2_add_test_root_macro(macro/makeTPCCCDBEntryForDCS.C
                       PUBLIC_LINK_LIBRARIES O2::DetectorsDCS
                                             O2::TPCdcs
                       LABELS tpc)

#if (OpenMP_CXX_FOUND)
    #target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    #target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
#endif()
