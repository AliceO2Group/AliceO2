# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_header_only_library(FrameworkFoundation)

o2_add_test(test_Traits NAME test_FrameworkFoundation_test_Traits
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_Traits.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)

o2_add_test(test_StructToTuple NAME test_FrameworkFoundation_StructToTuple
            COMPONENT_NAME FrameworkFoundation
            SOURCES test/test_StructToTuple.cxx
            PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation)
