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

#
# Print a list of the Root macros that exists in the repository but for which
# the o2_add_test_root_macro function has not been called.
#
function(o2_report_non_tested_macros)
  file(GLOB_RECURSE listOfMacros RELATIVE ${CMAKE_SOURCE_DIR} *.C)
  list(LENGTH listOfMacros nmacros)
  foreach(m ${listOfMacros})
    if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS)
      list(APPEND notTested ${m})
    endif()
    if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED)
      list(APPEND notTestedCompiled ${m})
    endif()
  endforeach()
  list(LENGTH notTested n)
  list(LENGTH notTestedCompiled nc)
  if(${n} GREATER 0)
    message(
      STATUS
        "WARNING : ${n}(L) and ${nc}(C) over ${nmacros} Root macros are NOT tested (L for loading, C for compilation) "
      )
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
      if(NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS_COMPILED
         OR NOT ${m} IN_LIST LIST_OF_ROOT_MACRO_TESTS)
        message(STATUS "[${loading}] [${compile}] ${m}")
      endif()
    endforeach()
    message(STATUS)
  endif()
endfunction()
