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

o2_add_library(TPCQC
               SOURCES src/PID.cxx
                       src/Tracking.cxx
                       src/Helpers.cxx
                       src/Clusters.cxx
                       src/Tracks.cxx
                       src/DCSPTemperature.cxx
                       src/SACs.cxx
                       src/IDCsVsSACs.cxx
               PUBLIC_LINK_LIBRARIES O2::TPCBase
                                     O2::DataFormatsTPC
                                     O2::GPUO2Interface
                                     O2::TPCCalibration)


o2_target_root_dictionary(TPCQC
                          HEADERS include/TPCQC/PID.h
                                  include/TPCQC/Tracking.h
                                  include/TPCQC/Helpers.h
                                  include/TPCQC/Clusters.h
                                  include/TPCQC/Tracks.h
                                  include/TPCQC/CalPadWrapper.h
                                  include/TPCQC/DCSPTemperature.h
                                  include/TPCQC/SACs.h
                                  include/TPCQC/IDCsVsSACs.h)

o2_add_test(PID
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_PID.cxx
            LABELS tpc)

o2_add_test(Clusters
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_Clusters.cxx
            LABELS tpc
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(Tracks
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_Tracks.cxx
            LABELS tpc)

o2_add_test(SACs
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_SACs.cxx
            LABELS tpc)

o2_add_test(IDCsVsSACs
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_IDCsVsSACs.cxx
            LABELS tpc)

o2_add_test_root_macro(macro/runPID.C
                       PUBLIC_LINK_LIBRARIES O2::TPCQC
                                             O2::DataFormatsTPC
                                             O2::TPCBase
                                             LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/runClusters.C
                       PUBLIC_LINK_LIBRARIES O2::TPCQC
                                             O2::DataFormatsTPC
                                             O2::TPCBase
                                             LABELS tpc COMPILE_ONLY)

o2_add_test_root_macro(macro/runTracks.C
                       PUBLIC_LINK_LIBRARIES O2::TPCQC
                                             O2::DataFormatsTPC
                                             O2::TPCBase
                                             LABELS tpc COMPILE_ONLY)
