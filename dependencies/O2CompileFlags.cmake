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

#
# o2_build_warning_flags() builds a list of warning flags from their names.
#
set(O2_ENABLE_WARNINGS "OFF")
if(DEFINED ENV{ALIBUILD_O2_WARNINGS})
  set(O2_ENABLE_WARNINGS "ON")
endif()

function(o2_build_warning_flags)
  cmake_parse_arguments(PARSE_ARGV 0 A "" "PREFIX;OUTPUTVARNAME" "WARNINGS")

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  list(TRANSFORM A_WARNINGS STRIP)
  list(TRANSFORM A_WARNINGS PREPEND ${A_PREFIX})
  string(JOIN " " OUTPUT ${A_WARNINGS})
  set(${A_OUTPUTVARNAME} ${OUTPUT} PARENT_SCOPE)

endfunction()

set(O2_C_WARNINGS "pointer-sign;override-init")
set(O2_CXX_WARNINGS "catch-value;pessimizing-move;reorder;delete-non-virtual-dtor;deprecated-copy;redundant-move")
set(O2_COMMON_WARNINGS "overloaded-virtual;address;bool-compare;bool-operation;\
char-subscripts;comment;enum-compare;format;format-overflow;\
format-truncation;int-in-bool-context;init-self;logical-not-parentheses;maybe-uninitialized;memset-elt-size;\
memset-transposed-args;misleading-indentation;missing-attributes;\
multistatement-macros;narrowing;nonnull;nonnull-compare;openmp-simd;parentheses;\
restrict;return-type;sequence-point;sign-compare;sizeof-pointer-div;\
sizeof-pointer-memaccess;strict-aliasing;strict-overflow;switch;tautological-compare;trigraphs;uninitialized;\
unused-label;unused-value;unused-variable;volatile-register-var;zero-length-bounds;\
unused-but-set-variable;stringop-truncation;clobbered;cast-function-type;\
empty-body;ignored-qualifiers;implicit-fallthrough;missing-field-initializers;sign-compare;\
string-compare;type-limits;uninitialized;shift-negative-value")


if(O2_ENABLE_WARNINGS)

o2_build_warning_flags(PREFIX "-W"
              OUTPUTVARNAME O2_C_ENABLED_WARNINGS
              WARNINGS ${O2_COMMON_WARNINGS} ${O2_C_WARNINGS} "array-bounds=1")
o2_build_warning_flags(PREFIX "-Wno-error="
              OUTPUTVARNAME O2_C_ENABLED_WARNINGS_NO_ERROR
              WARNINGS ${O2_COMMON_WARNINGS} ${O2_C_WARNINGS} "array-bounds")
o2_build_warning_flags(PREFIX "-W"
              OUTPUTVARNAME O2_CXX_ENABLED_WARNINGS
              WARNINGS ${O2_COMMON_WARNINGS} ${O2_CXX_WARNINGS} "array-bounds=1")
o2_build_warning_flags(PREFIX "-Wno-error="
              OUTPUTVARNAME O2_CXX_ENABLED_WARNINGS_NO_ERROR
              WARNINGS ${O2_COMMON_WARNINGS} ${O2_CXX_WARNINGS} "array-bounds")
else()
 message(STATUS "Building without compiler warnings enabled.")
endif()

string(JOIN " " CMAKE_C_WARNINGS "-Wno-unknown-warning-option" ${O2_C_ENABLED_WARNINGS} ${O2_C_ENABLED_WARNINGS_NO_ERROR})
string(JOIN " " CMAKE_CXX_WARNINGS "-Wno-unknown-warning-option" ${O2_CXX_ENABLED_WARNINGS} ${O2_CXX_ENABLED_WARNINGS_NO_ERROR})

set(CMAKE_CXX_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE}")
set(CMAKE_Fortran_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_LINK_FLAGS_COVERAGE "--coverage -fprofile-arcs  -fPIC")

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_Fortran_FLAGS_COVERAGE
    CMAKE_LINK_FLAGS_COVERAGE)

#Check the compiler and set the compile and link flags
IF (NOT CMAKE_BUILD_TYPE)
  Message(STATUS "Set BuildType to DEBUG")
  set(CMAKE_BUILD_TYPE Debug)
ENDIF (NOT CMAKE_BUILD_TYPE)

IF(ENABLE_CASSERT) #For the CI, we want to have <cassert> assertions enabled
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
ELSE()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
      set(FAIR_MIN_SEVERITY "detail")
    endif()
ENDIF()

IF(ENABLE_THREAD_SAFETY_ANALYSIS)
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -Werror=thread-safety -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS")
ENDIF()

set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g")
# make sure Debug build not optimized (does not seem to work without CACHE + FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_C_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)

set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} ${CMAKE_CXX_WARNINGS}")
set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}} ${CMAKE_C_WARNINGS}")

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error") # avoid undefined in our libs
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # avoid undefined in our libs
endif()

if(ENABLE_TIME_TRACE)
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -ftime-trace")
endif()

if(DEFINED ENV{O2_CXXFLAGS_OVERRIDE})
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} $ENV{O2_CXXFLAGS_OVERRIDE}")
  message(STATUS "Setting CXXFLAGS Override $ENV{O2_CXXFLAGS_OVERRIDE}")
endif()


message(STATUS "Using build type: ${CMAKE_BUILD_TYPE} - CXXFLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
