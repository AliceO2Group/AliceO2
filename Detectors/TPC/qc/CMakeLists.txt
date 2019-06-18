# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TPCQC
               SOURCES src/PID.cxx
               PUBLIC_LINK_LIBRARIES O2::TPCBase)



o2_target_root_dictionary(TPCQC
                          HEADERS include/TPCQC/PID.h)

o2_add_test(PID
            COMPONENT_NAME tpc
            PUBLIC_LINK_LIBRARIES O2::TPCQC
            SOURCES test/test_PID.cxx
            LABELS tpc)
