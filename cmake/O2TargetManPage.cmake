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

# Generate a man page
#
# Make sure we have nroff. If that is not the case we will not generate man
# pages
find_program(NROFF_FOUND nroff)

function(o2_target_man_page target)
  if(NOT NROFF_FOUND)
    return()
  endif()
  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        "NAME;SECTION"
                        "")

  # check the target exists
  if(NOT TARGET ${target})
    # try with out naming conventions
    set(baseTargetName ${target})
    o2_name_target(${baseTargetName} NAME target)
    if(NOT TARGET ${target})
      # not a library, maybe an executable ?
      o2_name_target(${baseTargetName} NAME target IS_EXE)
      if(NOT TARGET ${target})
        message(FATAL_ERROR "Target ${target} does not exist")
      endif()
    endif()
  endif()

  if(NOT A_SECTION)
    set(A_SECTION 1)
  endif()
  if(NOT A_NAME)
    message(
      FATAL_ERROR
        "You must provide the name of the input man file in doc/<name>.<section>.in"
      )
  endif()
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/doc/${A_NAME}.${A_SECTION}.in)
    message(
      FATAL_ERROR
        "Input file ${CMAKE_CURRENT_SOURCE_DIR}/doc/${A_NAME}.${A_SECTION}.in does not exist"
      )
  endif()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${A_NAME}.${A_SECTION}
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/doc/${A_NAME}.${A_SECTION}.in
    COMMAND nroff
            -Tascii
            -man
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/${A_NAME}.${A_SECTION}.in
            >
            ${CMAKE_CURRENT_BINARY_DIR}/${A_NAME}.${A_SECTION}
    VERBATIM)
  # the prefix man. for the target name avoids circular dependencies for the man
  # pages added at top level. Simply droping the dependency for those does not
  # invoke the custom command on all systems.
  set(CUSTOM_TARGET_NAME man.${A_NAME}.${A_SECTION})
  add_custom_target(${CUSTOM_TARGET_NAME}
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${A_NAME}.${A_SECTION})
  add_dependencies(${target} ${CUSTOM_TARGET_NAME})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${A_NAME}.${A_SECTION}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/man/man${A_SECTION})
endfunction()
