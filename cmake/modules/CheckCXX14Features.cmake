################################################################################
#    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    #
#                                                                              #
#              This software is distributed under the terms of the             # 
#         GNU Lesser General Public Licence version 3 (LGPL) version 3,        #  
#                  copied verbatim in the file "LICENSE"                       #
################################################################################

#blame: Mikolaj Krzewicki, mkrzewic@cern.ch
#based on the work by Rolf Eike Beer and Andreas Weis for the FairRoot project

# - Check which parts of the C++14 standard the compiler supports
#
# When found it will set the following variables
# one for each feature check (to be set at the end of the file when invoking tests)
#
#  HAS_CXX14_MAKE_UNIQUE                - make_unique support
#  HAS_CXX14_AGGREGATE-INITIALIZATION   - aggregate initialization support
#  HAS_CXX14_BINARY-LITERALS            - binary literals support
#  HAS_CXX14_GENERIC-LAMBDA             - generic lambdas support
#  HAS_CXX14_USER-DEFINED-LITERALS      - user defined literals support

#
# Each feature may have up to 3 checks, every one of them in it's own file
# FEATURE.cpp              - example that must build and return 0 when run
# FEATURE_fail.cpp         - example that must build, but may not return 0 when run
# FEATURE_fail_compile.cpp - example that must fail compilation
#
# The first one is mandatory, the latter 2 are optional and do not depend on
# each other (i.e. only one may be present).
#

if (NOT CMAKE_CXX_COMPILER_LOADED)
    message(FATAL_ERROR "CheckCXX14Features modules only works if language CXX is enabled")
endif ()

cmake_minimum_required(VERSION 2.8.2)

### Check for needed compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++14" _HAS_CXX14_FLAG)
if (NOT _HAS_CXX14_FLAG)
  message(FATAL_ERROR "Compiler does not support -std=c++14 option")
endif ()

function(cxx14_check_feature FEATURE_NAME RESULT_VAR)
    if (NOT DEFINED ${RESULT_VAR})
        set(_bindir "${CMAKE_CURRENT_BINARY_DIR}/cxx14/cxx14_${FEATURE_NAME}")

        set(_SRCFILE_BASE ${CheckCXX14SrcDir}/cxx14-test-${FEATURE_NAME})
        set(_LOG_NAME "\"${FEATURE_NAME}\"")
        message(STATUS "Checking C++14 support for ${_LOG_NAME}")

        set(_SRCFILE "${_SRCFILE_BASE}.cxx")
        set(_SRCFILE_FAIL "${_SRCFILE_BASE}_fail.cxx")
        set(_SRCFILE_FAIL_COMPILE "${_SRCFILE_BASE}_fail_compile.cxx")

        if (CROSS_COMPILING)
            try_compile(${RESULT_VAR} "${_bindir}" "${_SRCFILE}")
            if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
                try_compile(${RESULT_VAR} "${_bindir}_fail" "${_SRCFILE_FAIL}")
            endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        else (CROSS_COMPILING)
            try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
                    "${_bindir}" "${_SRCFILE}")
            if (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
                set(${RESULT_VAR} TRUE)
            else (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
                set(${RESULT_VAR} FALSE)
            endif (_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
            if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
                try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
                        "${_bindir}_fail" "${_SRCFILE_FAIL}")
                if (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
                    set(${RESULT_VAR} TRUE)
                else (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
                    set(${RESULT_VAR} FALSE)
                endif (_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
            endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        endif (CROSS_COMPILING)
        if (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL_COMPILE})
            try_compile(_TMP_RESULT "${_bindir}_fail_compile" "${_SRCFILE_FAIL_COMPILE}")
            if (_TMP_RESULT)
                set(${RESULT_VAR} FALSE)
            else (_TMP_RESULT)
                set(${RESULT_VAR} TRUE)
            endif (_TMP_RESULT)
        endif (${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL_COMPILE})

        if (${RESULT_VAR})
            message(STATUS "Checking C++14 support for ${_LOG_NAME}: works")
        else (${RESULT_VAR})
          message(FATAL_ERROR "Checking C++14 support for ${_LOG_NAME}: not supported")
        endif (${RESULT_VAR})
        set(${RESULT_VAR} ${${RESULT_VAR}} CACHE INTERNAL "C++14 support for ${_LOG_NAME}")
    endif (NOT DEFINED ${RESULT_VAR})
endfunction(cxx14_check_feature)

cxx14_check_feature("make_unique" HAS_CXX14_MAKE_UNIQUE)
cxx14_check_feature("aggregate-initialization" HAS_CXX14_AGGREGATE-INITIALIZATION)
cxx14_check_feature("binary-literals" HAS_CXX14_BINARY-LITERALS)
cxx14_check_feature("generic-lambda" HAS_CXX14_GENERIC-LAMBDA)
cxx14_check_feature("user-defined-literals" HAS_CXX14_USER-DEFINED-LITERALS)

