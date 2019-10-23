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
                       test/DPLBroadcasterMerger.cxx
                       test/DPLOutputTest.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework)

o2_add_executable(raw-proxy
                  COMPONENT_NAME dpl
                  SOURCES src/raw-proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils)

o2_add_test(DPLBroadcasterMerger
            SOURCES test/test_DPLBroadcasterMerger.cxx src/Utils.cxx
                    test/DPLBroadcasterMerger.cxx src/DPLMerger.cxx
                    src/DPLBroadcaster.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)

o2_add_test(DPLOutput
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
            SOURCES test/test_RootTreeReader.cxx
            PUBLIC_LINK_LIBRARIES O2::DPLUtils
            COMPONENT_NAME DPLUtils
            LABELS dplutils
            COMMAND_LINE_ARGS ${DPL_WORKFLOW_TESTS_EXTRA_OPTIONS} --run)
