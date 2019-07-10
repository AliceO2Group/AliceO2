# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(MemoryResources
               SOURCES src/MemoryResources.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

o2_add_test(MemoryResources
            SOURCES test/testMemoryResources.cxx
            PUBLIC_LINK_LIBRARIES O2::MemoryResources
            COMPONENT_NAME MemoryResources)

o2_add_test(observer_ptr
            SOURCES test/test_observer_ptr.cxx
            PUBLIC_LINK_LIBRARIES O2::MemoryResources
            COMPONENT_NAME MemoryResources)
