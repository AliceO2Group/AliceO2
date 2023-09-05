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
cmake_minimum_required(VERSION 3.5)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/Framework
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

o2_add_library(FrameworkFoundation
               SOURCES src/RuntimeError.cxx
               src/Signpost.cxx
               TARGETVARNAME targetName
               PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation3rdparty
              )
set(DPL_ENABLE_BACKTRACE ON CACHE BOOL "Enable backtrace on o2::framework::runtime_error")

if (DPL_ENABLE_BACKTRACE)
target_compile_definitions(${targetName} PUBLIC -DDPL_ENABLE_BACKTRACE)
endif()

add_executable(o2-test-framework-foundation
               test/test_FunctionalHelpers.cxx
               test/test_Traits.cxx
               test/test_StructToTuple.cxx
               test/test_CallbackRegistry.cxx
               test/test_CompilerBuiltins.cxx
               #               test/test_Signpost.cxx
               test/test_RuntimeError.cxx)
target_link_libraries(o2-test-framework-foundation PRIVATE O2::FrameworkFoundation)
target_link_libraries(o2-test-framework-foundation PRIVATE O2::Catch2)

add_executable(o2-test-framework-Signpost
               test/test_Signpost.cxx)
target_link_libraries(o2-test-framework-Signpost PRIVATE O2::FrameworkFoundation)

add_executable(o2-test-framework-SignpostLogger
               test/test_SignpostLogger.cxx
               test/test_SignpostLogger2.cxx
               )
target_link_libraries(o2-test-framework-SignpostLogger PRIVATE O2::FrameworkFoundation)

get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests ABSOLUTE)
set_property(TARGET o2-test-framework-foundation PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})
set_property(TARGET o2-test-framework-Signpost PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})
set_property(TARGET o2-test-framework-SignpostLogger PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})

add_test(NAME framework:foundation COMMAND o2-test-framework-foundation)

add_subdirectory(3rdparty)
