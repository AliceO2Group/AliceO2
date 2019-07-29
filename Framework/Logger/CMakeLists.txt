# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_header_only_library(FrameworkLogger
               INTERFACE_LINK_LIBRARIES fmt::fmt FairLogger::FairLogger)

# FIXME: the NAME parameter is just there to ease the comparison with previous
# test names, can be omitted later on
o2_add_test(Logger NAME test_FrameworkLogger_unittest_Logger
            SOURCES test/unittest_Logger.cxx
            COMPONENT_NAME framework
            PUBLIC_LINK_LIBRARIES O2::FrameworkLogger)
