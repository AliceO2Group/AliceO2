# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(Ex4
               SOURCES src/A.cxx src/B.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

o2_target_root_dictionary(Ex4 HEADERS include/Ex4/A.h src/B.h)

o2_add_executable(ex4
                  SOURCES src/run.cxx
                  PUBLIC_LINK_LIBRARIES O2::Ex4 O2::Ex2 O2::Ex3
                  COMPONENT_NAME example)

o2_add_test(test1
            SOURCES test/test1.cxx
            PUBLIC_LINK_LIBRARIES O2::Ex4
            COMPONENT_NAME Ex4
            LABELS fast dummy obvious
            INSTALL)

o2_add_test(test2
            SOURCES test/test2.cxx
            PUBLIC_LINK_LIBRARIES O2::Ex4 O2::Ex3 O2::Ex2
            COMPONENT_NAME Ex4
            LABELS fast dummy)
