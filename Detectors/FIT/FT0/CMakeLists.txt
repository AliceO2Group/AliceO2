# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

add_subdirectory(base)
add_subdirectory(raw)
add_subdirectory(reconstruction)
add_subdirectory(simulation)
add_subdirectory(workflow)

o2_add_test(ft0rawtest
            PUBLIC_LINK_LIBRARIES O2::FT0Simulation
            SOURCES test/test_raw_conversion_ft0.cxx
            COMPONENT_NAME ft0
            LABELS ft0)
