# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Preamble

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# it's important to specify accurately the list of languages. for instance C and
# C++ as we _do_ have some C files to compile explicitely as C (e.g. gl3w.c)
project(O2 LANGUAGES C CXX VERSION 1.2.0)

include(CTest)

# Project wide setup

# Would better fit inside GPU/CMakeLists.txt, but include GPU/Common directly
set(ALIGPU_BUILD_TYPE "O2")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY REPORT_UNDEFINED_PROPERTIES)
include(O2BuildSanityChecks)
o2_build_sanity_checks()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include(O2CheckCXXFeatures)
o2_check_cxx_features()

include(O2DefineOptions)
o2_define_options()

include(O2DefineOutputPaths)
o2_define_output_paths()

include(O2DefineRPATH)
o2_define_rpath()

# External dependencies
include(dependencies/CMakeLists.txt)

# include macros and functions that are used in the following subdirectories'
# CMakeLists.txt
include(O2AddExecutable)
include(O2AddHeaderOnlyLibrary)
include(O2AddLibrary)
include(O2AddTest)
include(O2AddTestRootMacro)
include(O2TargetRootDictionary)
include(O2DataFile)
include(O2TargetManPage)

# Main targets of the project in various subdirectories. Order matters.
add_subdirectory(Common/Constants)
add_subdirectory(Framework)
add_subdirectory(GPU/Common)
add_subdirectory(Common/MathUtils)
add_subdirectory(DataFormats/Detectors/Common)
add_subdirectory(DataFormats/common)
add_subdirectory(DataFormats/Reconstruction)
add_subdirectory(Common)
add_subdirectory(DataFormats/MemoryResources)
add_subdirectory(DataFormats/Headers)
add_subdirectory(Utilities/O2Device)
add_subdirectory(CCDB)
add_subdirectory(Algorithm)
add_subdirectory(DataFormats/Parameters)
add_subdirectory(Detectors/Base)
add_subdirectory(DataFormats/simulation)
add_subdirectory(Detectors/ITSMFT/common/base)
add_subdirectory(DataFormats/Detectors)
add_subdirectory(DataFormats/TimeFrame)
add_subdirectory(Utilities/PCG)
add_subdirectory(EventVisualisation)
add_subdirectory(Generators)
add_subdirectory(Detectors/TRD)
add_subdirectory(Detectors/ITSMFT)
add_subdirectory(GPU)
add_subdirectory(Detectors)
add_subdirectory(Utilities)
add_subdirectory(Steer) # consider building this only for simulation ?

if(BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

if(BUILD_SIMULATION)
  add_subdirectory(run)
endif()

add_subdirectory(config)

add_custom_target(man ALL)
o2_target_man_page(man NAME o2)
o2_target_man_page(man NAME FairMQDevice)

# Testing and packaging only needed if we are the top level directory
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Documentation
  add_subdirectory(doc)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
  if(BUILD_TEST_ROOT_MACROS)
    add_subdirectory(macro)
    include(O2ReportNonTestedMacros)
    o2_report_non_tested_macros()
  endif()
  add_subdirectory(packaging)

endif()
