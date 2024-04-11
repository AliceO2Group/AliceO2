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

string(REGEX MATCH "-O[0-9]+" CMAKE_FLAGS_OPT_VALUE "${CMAKE_CXX_FLAGS}")
if(NOT CMAKE_FLAGS_OPT_VALUE OR CMAKE_FLAGS_OPT_VALUE STREQUAL "-O0" OR CMAKE_FLAGS_OPT_VALUE STREQUAL "-O1")
  set(CMAKE_FLAGS_OPT_VALUE "-O2") # Release builds should not decrease the -O level below something requested externally via CXXFLAGS
endif()

set(CMAKE_CXX_FLAGS_COVERAGE "-g ${CMAKE_FLAGS_OPT_VALUE} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE}")
set(CMAKE_Fortran_FLAGS_COVERAGE "-g ${CMAKE_FLAGS_OPT_VALUE} -fprofile-arcs -ftest-coverage")
set(CMAKE_LINK_FLAGS_COVERAGE "--coverage -fprofile-arcs  -fPIC")

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_Fortran_FLAGS_COVERAGE
    CMAKE_LINK_FLAGS_COVERAGE)

# Options to enable the thread sanitizer
set(CMAKE_CXX_FLAGS_THREADSANITIZER "-g ${CMAKE_FLAGS_OPT_VALUE} -fsanitize=thread -fPIC")
set(CMAKE_C_FLAGS_THREADSANITIZER "${CMAKE_CXX_FLAGS_THREADSANITIZER}")
set(CMAKE_Fortran_FLAGS_THREADSANITIZER "-g ${CMAKE_FLAGS_OPT_VALUE} -fsanitize=thread -fPIC")
set(CMAKE_LINK_FLAGS_THREADSANITIZER "-fsanitize=thread -fPIC")

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_THREADSANITIZER
    CMAKE_C_FLAGS_THREADSANITIZER
    CMAKE_Fortran_FLAGS_THREADSANITIZER
    CMAKE_LINK_FLAGS_THREADSANITIZER)

IF(ENABLE_CASSERT) #For the CI, we want to have <cassert> assertions enabled
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_FLAGS_OPT_VALUE}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_FLAGS_OPT_VALUE} -g")
ELSE()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_FLAGS_OPT_VALUE} -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_FLAGS_OPT_VALUE} -g -DNDEBUG")
    if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
      set(FAIR_MIN_SEVERITY "detail")
    endif()
ENDIF()

IF(ENABLE_THREAD_SAFETY_ANALYSIS)
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -Werror=thread-safety -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS")
ENDIF()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_FLAGS_OPT_VALUE}")
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_FLAGS_OPT_VALUE}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_FLAGS_OPT_VALUE} -g")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_FLAGS_OPT_VALUE} -g")
# make sure Debug build not optimized (does not seem to work without CACHE + FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_C_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)

set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${CMAKE_CXX_WARNINGS}")
set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${CMAKE_C_WARNINGS}")

if(APPLE)
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # avoid undefined in our libs
endif()

if(ENABLE_TIME_TRACE)
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -ftime-trace")
endif()

if(DEFINED ENV{O2_CXXFLAGS_OVERRIDE})
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} $ENV{O2_CXXFLAGS_OVERRIDE}")
  message(STATUS "Setting CXXFLAGS Override $ENV{O2_CXXFLAGS_OVERRIDE}")
endif()

if(GPUCA_NO_FAST_MATH_WHOLEO2)
  set(GPUCA_NO_FAST_MATH 1)
  add_definitions(-DGPUCA_NO_FAST_MATH)
  set(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -fno-fast-math -ffp-contract=off")
  set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER} "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -fno-fast-math -ffp-contract=off")
endif()

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE} - CXXFLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
