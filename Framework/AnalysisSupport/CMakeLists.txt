# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Given GCC 7.3 does not provide std::filesystem we use Boost instead
# Drop this once we move to GCC 8.2+

if(TARGET JAliEn::JAliEn)
  set(EXTRA_TARGETS XRootD::Client JAliEn::JAliEn)
endif()

o2_add_library(FrameworkOnDemandTablesSupport
               SOURCES src/OnDemandPlugin.cxx
                       src/AODReaderHelpers.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework ${EXTRA_TARGETS})

o2_add_library(FrameworkAnalysisSupport
               SOURCES src/Plugin.cxx
                       src/DataInputDirector.cxx
                       src/TableTreeHelpers.cxx
                       src/AODJAlienReaderHelpers.cxx
                       src/AODWriterHelpers.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework ${EXTRA_TARGETS} ROOT::TreePlayer)

o2_add_library(FrameworkAnalysisRNTupleSupport
               SOURCES src/RNTuplePlugin.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework ${EXTRA_TARGETS} ROOT::ROOTNTuple ROOT::ROOTNTupleUtil)

o2_add_library(FrameworkAnalysisTTreeSupport
               SOURCES src/TTreePlugin.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework ${EXTRA_TARGETS} ROOT::TreePlayer)

o2_add_test(DataInputDirector NAME test_Framework_test_DataInputDirector
               SOURCES test/test_DataInputDirector.cxx
               COMPONENT_NAME Framework
               LABELS framework
               PUBLIC_LINK_LIBRARIES O2::FrameworkAnalysisSupport)

add_executable(o2-test-framework-analysis-support
               test/test_NavigateToLevel.cxx)
target_link_libraries(o2-test-framework-analysis-support PRIVATE O2::FrameworkAnalysisSupport O2::Catch2)

get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests ABSOLUTE)
set_property(TARGET o2-test-framework-analysis-support PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})

add_test(NAME framework:analysis-support COMMAND o2-test-framework-analysis-support)

o2_add_test(TableToTree NAME benchmark_TableToTree
               SOURCES test/benchmark_TableToTree.cxx
               COMPONENT_NAME Framework
               LABELS framework
               PUBLIC_LINK_LIBRARIES O2::FrameworkAnalysisSupport benchmark::benchmark)
