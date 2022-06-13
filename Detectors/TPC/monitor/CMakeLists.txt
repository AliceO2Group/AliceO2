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

o2_add_library(TPCMonitor
               SOURCES src/SimpleEventDisplay.cxx
                       src/SimpleEventDisplayGUI.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Gui O2::TPCBase O2::TPCCalibration)

o2_target_root_dictionary(TPCMonitor
                          HEADERS include/TPCMonitor/SimpleEventDisplay.h
                                  include/TPCMonitor/SimpleEventDisplayGUI.h)

o2_add_executable(monitor
                  COMPONENT_NAME tpc
                  SOURCES run/runMonitor.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCMonitor ROOT::Rint)

o2_add_test_root_macro(macro/RunCompareMode3.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)
o2_add_test_root_macro(macro/RunFindAdcError.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)
