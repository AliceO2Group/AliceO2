# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(TimeFrame
               SOURCES src/TimeFrame.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ O2::Headers ROOT::RIO)

o2_target_root_dictionary(TimeFrame HEADERS include/TimeFrame/TimeFrame.h)

o2_add_test(TimeFrameTest
            SOURCES test/TimeFrameTest.cxx
            PUBLIC_LINK_LIBRARIES O2::TimeFrame
            COMPONENT_NAME DataFormats)
