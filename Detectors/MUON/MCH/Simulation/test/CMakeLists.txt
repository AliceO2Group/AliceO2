# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(digit-merging
        COMPONENT_NAME mch
        SOURCES testDigitMerging.cxx DigitMerging.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase O2::DetectorsPassive
        LABELS muon mch sim)

o2_add_test(digitizer
        COMPONENT_NAME mch
        SOURCES testDigitizer.cxx
        PUBLIC_LINK_LIBRARIES O2::DetectorsPassive
                              O2::MCHBase
                              O2::MCHGeometryTest
                              O2::MCHSimulation
        LABELS muon mch long sim)

# o2_add_test(response
#         COMPONENT_NAME mch
#         SOURCES testResponse.cxx
#         PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase O2::DetectorsPassive
#         LABELS muon mch  sim)

o2_add_test(regular-geometry
        COMPONENT_NAME mch
        SOURCES
        testRegularGeometry.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHSimulation O2::MCHBase O2::DetectorsPassive
        LABELS muon mch sim)

if(benchmark_FOUND)
        o2_add_executable(
                digit-merging
                COMPONENT_NAME mch
                SOURCES benchDigitMerging.cxx DigitMerging.cxx
                PUBLIC_LINK_LIBRARIES O2::MCHSimulation benchmark::benchmark)

endif()
