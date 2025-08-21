# Copyright 2019-2025 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
o2_add_library(FrameworkCCDBSupport
               SOURCES
                src/Plugin.cxx
                src/CCDBFetcherHelper.cxx
                src/CCDBHelpers.cxx
                src/AnalysisCCDBHelpers.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework O2::CCDB)

add_executable(o2-test-framework-ccdbsupport
               test/test_CCDBHelpers.cxx)
target_link_libraries(o2-test-framework-ccdbsupport PRIVATE O2::Framework)
target_link_libraries(o2-test-framework-ccdbsupport PRIVATE O2::FrameworkCCDBSupport)
target_link_libraries(o2-test-framework-ccdbsupport PRIVATE O2::Catch2)

get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests ABSOLUTE)
set_property(TARGET o2-test-framework-ccdbsupport PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})

add_test(NAME framework:ccdbsupport COMMAND o2-test-framework-ccdbsupport --skip-benchmarks)
