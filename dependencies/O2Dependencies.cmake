# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

include("${CMAKE_CURRENT_LIST_DIR}/O2RecipeAdapter.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_MODULE_PATH})

if(ALIBUILD_BASEDIR)
  # try autodetecting external packages from an alibuild installation zone
  include(O2FindDependenciesFromAliBuild)
  o2_find_dependencies_from_alibuild(${ALIBUILD_BASEDIR} LABEL ${ALIBUILD_LABEL}
                                     QUIET)
endif()

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

find_package(arrow CONFIG)
set_package_properties(arrow PROPERTIES TYPE REQUIRED)

find_package(Vc)
set_package_properties(Vc PROPERTIES TYPE REQUIRED)

find_package(ROOT 6.16.00 MODULE)
set_package_properties(ROOT PROPERTIES TYPE REQUIRED)

find_package(fmt)
set_package_properties(fmt PROPERTIES TYPE REQUIRED)

find_package(Boost 1.59
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

find_package(ms_gsl MODULE)
set_package_properties(ms_gsl
                       PROPERTIES
                       TYPE REQUIRED
                       PURPOSE "Mainly for its span")

find_package(FairMQ CONFIG)
set_package_properties(FairMQ PROPERTIES TYPE REQUIRED)

find_package(protobuf CONFIG)
set_package_properties(protobuf PROPERTIES TYPE REQUIRED PURPOSE "For CCDB API")

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

# MC specific packages
message(STATUS "Input BUILD_SIMULATION=${BUILD_SIMULATION}")
include("${CMAKE_CURRENT_LIST_DIR}/O2SimulationDependencies.cmake")
message(STATUS "Output BUILD_SIMULATION=${BUILD_SIMULATION}")

# Optional packages

find_package(DDS CONFIG)
set_package_properties(DDS PROPERTIES TYPE RECOMMENDED)
find_package(benchmark CONFIG NAMES benchmark googlebenchmark)
set_package_properties(benchmark PROPERTIES TYPE OPTIONAL)
find_package(OpenMP)
set_package_properties(OpenMP PROPERTIES TYPE OPTIONAL)
find_package(GLFW NAMES glfw3 CONFIG)
set_package_properties(GLFW PROPERTIES TYPE RECOMMENDED)
find_package(AliRoot)
set_package_properties(AliRoot
                       PROPERTIES
                       TYPE OPTIONAL
                       PURPOSE "For very specific use cases only")

find_package(GLEW)
set_package_properties(GLEW PROPERTIES TYPE OPTIONAL)

find_package(OpenGL)
set_package_properties(OpenGL PROPERTIES TYPE OPTIONAL)

find_package(LLVM)
set_package_properties(LLVM PROPERTIES TYPE OPTIONAL)
if(LLVM_FOUND)
find_package(Clang)
set_package_properties(Clang PROPERTIES TYPE OPTIONAL)
endif()


find_package(O2GPU)

feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)

