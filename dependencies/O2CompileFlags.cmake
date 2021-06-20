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

# Define the C, CXX, Fortran (and possibly linker) flags
# for the different build types we support :
# Debug, Release, RelWithDebInfo
# (FIXME: Coverage is left here but status unclear, to be reviewed ?)
#

set(CMAKE_CXX_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE}")
set(CMAKE_Fortran_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_LINK_FLAGS_COVERAGE "--coverage -fprofile-arcs  -fPIC")

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_Fortran_FLAGS_COVERAGE
    CMAKE_LINK_FLAGS_COVERAGE)

# Check the compiler and set the compile and link flags
#
# FIXME: review our usages of CMAKE_BUILD_TYPE, in particular wrt multi-config
# build systems (the only one we might use is Ninja Multi Config, as
# XCode and MSVC are less likely ;-), but still)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Set BuildType to DEBUG")
  set(CMAKE_BUILD_TYPE Debug)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
  set(FAIR_MIN_SEVERITY "info")
endif()

set(NDEBUG "-DNDEBUG")

IF(ENABLE_CASSERT)
# For the CI, we want to have <cassert> assertions enabled
  set(NDEBUG "")
endif()

#
# Define compilers flags
#
# Note that we explicitely _append_ to the flags, to let the developper
# the option to change them from the outside,
# without having to mess with this file (or other CMakeLists.txt).
# That comes handy when using e.g. sanitizers
#

# Release flags

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 ${NDEBUG}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 ${NDEBUG}")
set(CMAKE_Fortran_FLAGS_RELEASE "-O2")

# RelWithDebInfo flags

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g ${NDEBUG}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -g ${NDEBUG}")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g")

# Debug flags

# FIXME: FORCE is most probably wrong here (see "Professional CMake :
# A Practical Guide, 15.4.2)
#
# make sure Debug build not optimized (does not seem to work without CACHE + FORCE)
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Debug mode build flags" FORCE)
# set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -g -O0")

# Link flags, for all build types

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error") # avoid undefined in our libs
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # avoid undefined in our libs
endif()

####

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
message(STATUS "Fortran_FLAGS: ${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE}}")

