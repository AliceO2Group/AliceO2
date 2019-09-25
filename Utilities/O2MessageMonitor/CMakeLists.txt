# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MessageMonitor
               SOURCES src/O2MessageMonitor.cxx
               PUBLIC_LINK_LIBRARIES FairLogger::FairLogger FairMQ::FairMQ
                                     O2::Device)

o2_add_executable(message-monitor
                  SOURCES src/runO2MessageMonitor.cxx
                  PUBLIC_LINK_LIBRARIES O2::MessageMonitor)

o2_add_test(O2MessageMonitorTest
            SOURCES test/O2MessageMonitorTest.cxx
            PUBLIC_LINK_LIBRARIES O2::MessageMonitor
            LABELS utils)
