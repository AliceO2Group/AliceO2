# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(DPLUtils
               SOURCES src/Utils.cxx
                       src/DPLBroadcaster.cxx
                       src/DPLGatherer.cxx
                       src/DPLMerger.cxx
                       src/DPLRouter.cxx
                       src/RawParser.cxx
                       test/DPLBroadcasterMerger.cxx
                       test/DPLOutputTest.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_executable(raw-proxy
                  COMPONENT_NAME dpl
                  SOURCES src/raw-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils)

o2_add_executable(raw-parser
                  COMPONENT_NAME dpl
                  SOURCES src/raw-parser.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils)

o2_add_executable(output-proxy
                  COMPONENT_NAME dpl
                  SOURCES src/dpl-output-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils)

o2_add_test(DPLBroadcasterMerger
            SOURCES test/test_DPLBroadcasterMerger.cxx src/Utils.cxx
                    test/DPLBroadcasterMerger.cxx src/DPLMerger.cxx
                    src/DPLBroadcaster.cxx
            NO_BOOST_TEST
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)

o2_add_test(DPLOutput
            NO_BOOST_TEST
            SOURCES test/test_DPLOutputTest.cxx src/Utils.cxx
                    test/DPLOutputTest.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS long dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)

o2_add_test(RootTreeWriter
            SOURCES test/test_RootTreeWriter.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils)

o2_add_test(RootTreeWriterWorkflow
            NO_BOOST_TEST
            SOURCES test/test_RootTreeWriterWorkflow.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)

o2_add_test(RootTreeReader
            NO_BOOST_TEST
            SOURCES test/test_RootTreeReader.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)

o2_add_test(RawParser
            SOURCES test/test_RawParser.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils)

o2_add_test(DPLRawParser
            SOURCES test/test_DPLRawParser.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils)

if (TARGET benchmark::benchmark)
foreach(b
        RawParser
        )
  o2_add_test(benchmark_${b} NAME test_Framework_benchmark_${b}
              SOURCES test/benchmark_${b}.cxx
              COMPONENT_NAME DPLUtils
              LABELS dplutils benchmark
              PUBLIC_LINK_LIBRARIES O2::DPLUtils benchmark::benchmark)
endforeach()
endif()
