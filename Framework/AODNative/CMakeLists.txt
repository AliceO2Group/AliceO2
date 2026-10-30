# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# ROOT-free reader for AO2D files. Nothing in this directory may depend on
# ROOT, at build or run time: see the check at the end of this file.
o2_add_library(AODNative
               SOURCES src/ByteSource.cxx
                       src/LocalByteSource.cxx
                       src/Container.cxx
                       src/Objects.cxx
                       src/Codecs.cxx
                       src/Decoder.cxx
                       src/ArrowTable.cxx
                       src/ReadPlan.cxx
               TARGETVARNAME targetName
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES Arrow::arrow_shared
               PRIVATE_LINK_LIBRARIES PkgConfig::zstd PkgConfig::lz4 PkgConfig::lzma ZLIB::ZLIB)

# Reading over root:// needs the XRootD client, which does not depend on ROOT.
if(XROOTD_FOUND AND XROOTD_CLIENT_FOUND)
  target_sources(${targetName} PRIVATE src/XRootDByteSource.cxx)
  target_include_directories(${targetName} SYSTEM PRIVATE ${XROOTD_INCLUDE_DIRS})
  target_link_libraries(${targetName} PRIVATE ${XROOTD_CLIENT_LIBRARIES})
  target_compile_definitions(${targetName} PRIVATE AODNATIVE_WITH_XROOTD)
endif()

# Public HTTP(S) with libcurl.
if(TARGET CURL::libcurl)
  target_sources(${targetName} PRIVATE src/HttpByteSource.cxx)
  target_link_libraries(${targetName} PRIVATE CURL::libcurl)
  target_compile_definitions(${targetName} PRIVATE AODNATIVE_WITH_CURL)
endif()

o2_add_executable(inspect
                  COMPONENT_NAME aod-native
                  SOURCES tools/inspect.cxx
                  PUBLIC_LINK_LIBRARIES O2::AODNative fmt::fmt)

o2_add_executable(dump
                  COMPONENT_NAME aod-native
                  SOURCES tools/dump.cxx
                  PUBLIC_LINK_LIBRARIES O2::AODNative)

add_executable(o2-test-framework-aod-native
               test/test_Codecs.cxx
               test/test_Objects.cxx
               test/test_Decoder.cxx
               test/test_ReadPlan.cxx)
target_include_directories(o2-test-framework-aod-native PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(o2-test-framework-aod-native PRIVATE O2::AODNative O2::Catch2 PkgConfig::zstd PkgConfig::lz4 PkgConfig::lzma ZLIB::ZLIB)

get_filename_component(outdir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../tests ABSOLUTE)
set_property(TARGET o2-test-framework-aod-native PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})

add_test(NAME framework:aod-native COMMAND o2-test-framework-aod-native)

# Measures private memory while decoding the files given on the command line.
# It interposes glibc's malloc, so it is Linux only and cannot be built with sanitizers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(o2-test-framework-aod-native-memory test/test_Memory.cxx)
  target_link_libraries(o2-test-framework-aod-native-memory PRIVATE O2::AODNative)
  set_property(TARGET o2-test-framework-aod-native-memory PROPERTY RUNTIME_OUTPUT_DIRECTORY ${outdir})
endif()

# libFuzzer target for the container, object and decoder layers. Needs clang,
# and the whole library built with -fsanitize=address,undefined to be useful.
option(AODNATIVE_FUZZ "Build the AODNative libFuzzer target" OFF)
if(AODNATIVE_FUZZ)
  add_executable(o2-fuzz-aod-native test/fuzz/fuzz_RootFile.cxx)
  target_include_directories(o2-fuzz-aod-native PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
  target_compile_options(o2-fuzz-aod-native PRIVATE -fsanitize=fuzzer)
  target_link_options(o2-fuzz-aod-native PRIVATE -fsanitize=fuzzer)
  target_link_libraries(o2-fuzz-aod-native PRIVATE O2::AODNative PkgConfig::zstd PkgConfig::lz4 PkgConfig::lzma ZLIB::ZLIB)
endif()

# The reader core must not link ROOT, directly or through its dependencies.
foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
  get_target_property(libs ${targetName} ${property})
  foreach(lib IN LISTS libs)
    if(lib MATCHES "ROOT::" OR lib MATCHES "O2::Framework")
      message(FATAL_ERROR "O2::AODNative must not depend on ROOT, but links ${lib}")
    endif()
  endforeach()
endforeach()
