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

include(O2RootMacroExclusionList)
include(O2GetListOfMacros)

#
# Loop over the Root macros that exists in the repository and check whether
# there is one (or two) test for that macro (i.e. the o2_add_test_root_macro
# function has been called). If the macro is not tested and is not part of a
# known exclusion list (defined in O2RootMacroExclusionList.cmake file), then a
# FATAL_ERROR is issued
#
# (unless the DO_NOT_FAIL parameter is given, in which case those cases are
# simply reported).
#
function(o2_report_non_tested_macros)

  cmake_parse_arguments(PARSE_ARGV
                        0
                        A
                        "DO_NOT_FAIL;QUIET"
                        ""
                        "")

  o2_get_list_of_macros(${CMAKE_SOURCE_DIR} listOfMacros)

  list(LENGTH listOfMacros nmacros)
  foreach(m ${listOfMacros})
    if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS)
      list(APPEND notTestedLoading ${m})
    endif()
    if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED)
      list(APPEND notTestedCompiled ${m})
    endif()
  endforeach()
  list(LENGTH notTestedLoading n)
  list(LENGTH notTestedCompiled nc)
  list(LENGTH O2_ROOT_MACRO_EXCLUSION_LIST ne)
  if(${n} GREATER 0)
    if(NOT A_QUIET)
      message(
        STATUS
          "WARNING : ${n}(L) and ${nc}(C) over ${nmacros} Root macros are NOT tested (L for loading, C for compilation). ${ne} are exempted (E)."
        )
    endif()
    message(STATUS)
    foreach(m ${listOfMacros})
      set(loading " ")
      set(compile " ")
      if(${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS)
        set(loading "L")
      endif()
      if(${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED)
        set(compile "C")
      endif()
      if(${m} IN_LIST O2_ROOT_MACRO_EXCLUSION_LIST)
        set(exempted "E")
      endif()
      if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED
         OR NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS
         OR ${m} IN_LIST O2_ROOT_MACRO_EXCLUSION_LIST)
        if(NOT A_QUIET)
          message(STATUS "[${loading}] [${compile}] [${exempted}] ${m}")
        endif()
      endif()
      if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS
         AND NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED
         AND NOT ${m} IN_LIST O2_ROOT_MACRO_EXCLUSION_LIST)
        if(NOT A_DO_NOT_FAIL)
          message(FATAL_ERROR "Macro ${m} should be tested")
        else()
          message(WARNING "Macro ${m} is not tested at all !!!")
        endif()
      endif()
    endforeach()
    message(STATUS)
  endif()
endfunction()
