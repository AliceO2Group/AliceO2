# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/ for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FrameworkLogger
               SOURCES src/Logger.cxx
               PUBLIC_LINK_LIBRARIES FairLogger::FairLogger) # FIXME: should be
                                                             # changed to {fmt}
                                                             # once we have that
                                                             # as a dependency

o2_add_test(Logger
            SOURCES test/unittest_Logger.cxx
            COMPONENT_NAME framework
            PUBLIC_LINK_LIBRARIES O2::FrameworkLogger)
