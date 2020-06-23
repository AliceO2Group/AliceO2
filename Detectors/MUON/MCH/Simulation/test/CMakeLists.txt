# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(geometry-standalone
        NAME o2-test-mch-geometry-standalone
        COMPONENT_NAME mch
        SOURCES testGeometry.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase
        COMMAND_LINE_ARGS standalone
        LABELS muon mch sim)

o2_add_test(geometry-regular
        NAME o2-test-mch-geometry-regular
        COMPONENT_NAME mch
        SOURCES testGeometry.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase
        COMMAND_LINE_ARGS regular
        LABELS muon mch sim)

o2_add_test(simulation
        COMPONENT_NAME mch
        SOURCES testDigitMerging.cxx DigitMerging.cxx
        testDigitization.cxx testResponse.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase
        LABELS muon mch long sim)

if(benchmark_FOUND)
        o2_add_executable(
                digit-merging
                COMPONENT_NAME mch
                SOURCES benchDigitMerging.cxx DigitMerging.cxx
                PUBLIC_LINK_LIBRARIES O2::MCHSimulation benchmark::benchmark)

endif()
