# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# trying to get only one test exe for this module
o2_add_test(StressTest
            SOURCES src/CathodeSegmentation.cxx src/CathodeSegmentationLong.cxx
                    src/Segmentation.cxx
            COMPONENT_NAME mchmapping
            PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl3 O2::MCHMappingSegContour3
                                  RapidJSON::RapidJSON
            LABELS "muon;mch;long" CONFIGURATIONS RelWithDebInfo)

if(benchmark_FOUND)
  o2_add_executable(segmentation3
                    SOURCES src/BenchCathodeSegmentation.cxx
                            src/BenchSegmentation.cxx
                    IS_BENCHMARK
                    COMPONENT_NAME mch
                    PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl3
                                          O2::MCHMappingSegContour3
                                          benchmark::benchmark)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test_random_pos.json
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
