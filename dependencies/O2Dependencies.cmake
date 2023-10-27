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

include_guard()

include("${CMAKE_CURRENT_LIST_DIR}/O2TestsAdapter.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_MODULE_PATH})

# Required packages
#
# Order is not completely irrelevant. For instance arrow must come before
# FairRoot (see FindFairRoot.cmake)
#
# Generally speaking we should prefer the CONFIG variant of the find_package. We
# explicitely don't use the CONFIG variant (i.e. we do use the MODULE variant)
# only for some packages XXX where we define our own FindXXX.cmake module (e.g.
# to complement and/or fix what's done in the package's XXXConfig.cmake file)

include(FeatureSummary)

include(FindThreads)

find_package(Arrow CONFIG)
if(${Arrow_VERSION} VERSION_LESS 11)
find_package(Gandiva CONFIG PATHS ${Arrow_DIR} QUIET)
else()
find_package(Gandiva CONFIG)
endif()
set_package_properties(Arrow PROPERTIES TYPE REQUIRED)
set_package_properties(Gandiva PROPERTIES TYPE REQUIRED)

if (NOT TARGET Arrow::arrow_shared)
 add_library(Arrow::arrow_shared ALIAS arrow_shared)
endif()

if (NOT TARGET Gandiva::gandiva_shared)
  add_library(Gandiva::gandiva_shared ALIAS gandiva_shared)
endif()

find_package(Vc)
set_package_properties(Vc PROPERTIES TYPE REQUIRED)

find_package(ROOT 6.20.02)
set_package_properties(ROOT PROPERTIES TYPE REQUIRED)

find_package(VMC MODULE)

find_package(fmt)
set_package_properties(fmt PROPERTIES TYPE REQUIRED)

find_package(Boost 1.70
             COMPONENTS container
                        thread
                        system
                        timer
                        program_options
                        random
                        filesystem
                        chrono
                        exception
                        regex
                        serialization
                        log
                        log_setup
                        unit_test_framework
                        date_time
                        iostreams)
set_package_properties(Boost PROPERTIES TYPE REQUIRED)

find_package(FairLogger CONFIG)
set_package_properties(FairLogger PROPERTIES TYPE REQUIRED)

find_package(FairRoot MODULE)
set_package_properties(FairRoot PROPERTIES TYPE REQUIRED)

find_package(Microsoft.GSL CONFIG)
set_package_properties(Microsoft.GSL
                       PROPERTIES
                       TYPE REQUIRED
                       PURPOSE "Mainly for its span")

find_package(FairMQ CONFIG)
set_package_properties(FairMQ PROPERTIES TYPE REQUIRED)

# find_package(protobuf CONFIG)
# set_package_properties(protobuf PROPERTIES TYPE REQUIRED PURPOSE "For CCDB API")

find_package(InfoLogger CONFIG NAMES InfoLogger libInfoLogger)
set_package_properties(InfoLogger PROPERTIES TYPE REQUIRED)

find_package(Configuration CONFIG)
set_package_properties(Configuration PROPERTIES TYPE REQUIRED)

find_package(Monitoring CONFIG)
set_package_properties(Monitoring PROPERTIES TYPE REQUIRED)

find_package(Common CONFIG)
set_package_properties(Common PROPERTIES TYPE REQUIRED)

find_package(RapidJSON MODULE)
set_package_properties(RapidJSON PROPERTIES TYPE REQUIRED)

find_package(CURL)
set_package_properties(CURL PROPERTIES TYPE REQUIRED)

find_package(TBB)
set_package_properties(TBB PROPERTIES TYPE REQUIRED)

find_package(JAliEnROOT MODULE)
set_package_properties(JAliEnROOT PROPERTIES TYPE RECOMMENDED)

find_package(XRootD MODULE)
set_package_properties(XRootD PROPERTIES TYPE RECOMMENDED)

find_package(libjalienO2 MODULE)
set_package_properties(libjalienO2 PROPERTIES TYPE REQUIRED PURPOSE "For CCDB API")

# MC specific packages
message(STATUS "Input BUILD_SIMULATION=${BUILD_SIMULATION}")
include("${CMAKE_CURRENT_LIST_DIR}/O2SimulationDependencies.cmake")
message(STATUS "Output BUILD_SIMULATION=${BUILD_SIMULATION}")

# Optional packages

find_package(benchmark CONFIG NAMES benchmark googlebenchmark)
set_package_properties(benchmark PROPERTIES TYPE OPTIONAL)
find_package(OpenMP)
set_package_properties(OpenMP PROPERTIES TYPE OPTIONAL)
if (NOT OpenMP_CXX_FOUND AND CMAKE_SYSTEM_NAME MATCHES Darwin)
  message(STATUS "MacOS OpenMP not found, attempting workaround")
  find_package(OpenMPMacOS)
endif()

find_package(LibUV MODULE)
set_package_properties(LibUV PROPERTIES TYPE REQUIRED)
find_package(GLFW MODULE)
set_package_properties(GLFW PROPERTIES TYPE RECOMMENDED)
find_package(DebugGUI CONFIG)
set_package_properties(DebugGUI PROPERTIES TYPE RECOMMENDED)

find_package(AliRoot)
set_package_properties(AliRoot
                       PROPERTIES
                       TYPE OPTIONAL
                       PURPOSE "For very specific use cases only")

find_package(OpenGL)
set_package_properties(OpenGL PROPERTIES TYPE OPTIONAL)

find_package(LLVM)
set_package_properties(LLVM PROPERTIES TYPE OPTIONAL)
if(LLVM_FOUND)
find_package(Clang)
set_package_properties(Clang PROPERTIES TYPE OPTIONAL)
endif()


find_package(O2GPU)

find_package(FastJet)

find_package(FFTW3f CONFIG)
set_package_properties(FFTW3f PROPERTIES TYPE REQUIRED)

find_package(absl CONFIG)
set_package_properties(absl PROPERTIES TYPE REQUIRED)

find_package(Vtune)
set_package_properties(Vtune PROPERTIES TYPE OPTIONAL)

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)
