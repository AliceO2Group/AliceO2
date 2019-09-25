# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCMonitor
               SOURCES src/SimpleEventDisplay.cxx
               PUBLIC_LINK_LIBRARIES O2::TPCBase O2::TPCCalibration)

o2_target_root_dictionary(TPCMonitor
                          HEADERS include/TPCMonitor/SimpleEventDisplay.h)

o2_add_executable(monitor
                  COMPONENT_NAME tpc
                  SOURCES run/runMonitor.cxx
                  PUBLIC_LINK_LIBRARIES O2::TPCMonitor)

o2_add_test_root_macro(macro/RunCompareMode3.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)
o2_add_test_root_macro(macro/RunFindAdcError.C
                       PUBLIC_LINK_LIBRARIES O2::TPCReconstruction
                       LABELS tpc)
o2_add_test_root_macro(macro/RunSimpleEventDisplay.C
                       PUBLIC_LINK_LIBRARIES O2::TPCMonitor
                       LABELS tpc
                       ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
