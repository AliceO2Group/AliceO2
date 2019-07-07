# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(Mapping
            SOURCES src/testMapping.cxx
            COMPONENT_NAME mid
            PUBLIC_LINK_LIBRARIES O2::MIDBase
            LABELS muon mid)

o2_add_test(Positions
            SOURCES src/Positions.cxx
            COMPONENT_NAME mid
            PUBLIC_LINK_LIBRARIES RapidJSON::RapidJSON O2::MIDBase
            LABELS muon mid)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
