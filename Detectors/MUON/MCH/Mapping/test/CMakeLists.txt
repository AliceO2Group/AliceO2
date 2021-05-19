# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

foreach(
  impl
  RANGE 3
        4)

  o2_add_executable(
    generate-pad-indices-impl${impl}
    SOURCES src/generatePadIndices.cxx
    COMPONENT_NAME mchmapping
    PUBLIC_LINK_LIBRARIES
      O2::MCHMappingImpl${impl}
      RapidJSON::RapidJSON
      Boost::program_options)

  o2_add_test(
    PadIndices${impl}
    NAME o2-test-mchmapping-pad-indices-impl${impl}
    SOURCES src/testPadIndices.cxx
    COMPONENT_NAME mchmapping
    MAX_ATTEMPTS 1
    COMMAND_LINE_ARGS
      --filepattern
      ${CMAKE_CURRENT_LIST_DIR}/data/test_pad_indices_de{}.json
      --de
      100
      --de
      300
      --de
      500
      --de
      501
      --de
      502
      --de
      503
      --de
      504
      --de
      600
      --de
      601
      --de
      602
      --de
      700
      --de
      701
      --de
      702
      --de
      703
      --de
      704
      --de
      705
      --de
      706
      --de
      902
      --de
      903
      --de
      904
      --de
      905
    PUBLIC_LINK_LIBRARIES
      O2::MCHMappingImpl${impl}
      O2::MCHMappingSegContour
      RapidJSON::RapidJSON
      Boost::program_options
    NO_BOOST_TEST
    LABELS "muon;mch;long")

endforeach()

o2_add_test(
  StressTest3
  NAME o2-test-mchmapping-stress-test-impl3
  SOURCES src/CathodeSegmentation.cxx
          src/CathodeSegmentationLong.cxx
          src/Segmentation.cxx
          src/TestParameters.cxx
  COMPONENT_NAME mchmapping
  MAX_ATTEMPTS 1
  COMMAND_LINE_ARGS
    --testpos
    ${CMAKE_CURRENT_LIST_DIR}/data/test_random_pos.json
    --run2
    --manunumbering
  PUBLIC_LINK_LIBRARIES
    O2::MCHMappingImpl3
    O2::MCHMappingSegContour
    RapidJSON::RapidJSON
  CONFIGURATIONS
    RelWithDebInfo
    Release
    MinSizeRel
  LABELS "muon;mch;long")

o2_add_test(
  StressTest4
  NAME o2-test-mchmapping-stress-test-impl4
  SOURCES src/CathodeSegmentation.cxx
          src/CathodeSegmentationLong.cxx
          src/Segmentation.cxx
          src/TestParameters.cxx
  COMPONENT_NAME mchmapping
  MAX_ATTEMPTS 1
  COMMAND_LINE_ARGS
    --testpos
    ${CMAKE_CURRENT_LIST_DIR}/data/test_random_pos.json
    --manunumbering
  PUBLIC_LINK_LIBRARIES
    O2::MCHMappingImpl4
    O2::MCHMappingSegContour
    RapidJSON::RapidJSON
  CONFIGURATIONS
    RelWithDebInfo
    Release
    MinSizeRel
  LABELS "muon;mch;long")

if(benchmark_FOUND)
  o2_add_executable(
    segmentation4
    SOURCES src/BenchCathodeSegmentation.cxx
            src/BenchSegmentation.cxx
    IS_BENCHMARK
    COMPONENT_NAME mch
    PUBLIC_LINK_LIBRARIES
      O2::MCHMappingImpl4
      O2::MCHMappingSegContour
      benchmark::benchmark)
endif()
