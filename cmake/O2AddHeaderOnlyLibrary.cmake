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

# o2_add_header_only_library creates a header-only target.
#
# * INCLUDE_DIRECTORIES the relative path(s) to the headers of that library if
#   not specified will be set as "include" simply (which should work just fine
#   in most cases)
#
function(o2_add_header_only_library baseTargetName)

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        ""
                        "INCLUDE_DIRECTORIES;INTERFACE_LINK_LIBRARIES")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  o2_name_target(${baseTargetName} NAME target)

  # define the target and its O2:: alias
  add_library(${target} INTERFACE)
  add_library(O2::${baseTargetName} ALIAS ${target})

  # set the export name so that packages using O2 can reference the target as
  # O2::${baseTargetName} as well (assuming the export is installed with
  # namespace O2::)
  set_property(TARGET ${target} PROPERTY EXPORT_NAME ${baseTargetName})

  if(NOT A_INCLUDE_DIRECTORIES)
    get_filename_component(dir include ABSOLUTE)
    if(EXISTS ${dir})
      set(A_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${dir}>)
    else()
      set(A_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
    endif()
  endif()

  target_include_directories(
    ${target}
    INTERFACE $<BUILD_INTERFACE:${A_INCLUDE_DIRECTORIES}>)

  if(A_INTERFACE_LINK_LIBRARIES)
    target_link_libraries(${target} INTERFACE ${A_INTERFACE_LINK_LIBRARIES})
  endif()
  install(DIRECTORY ${A_INCLUDE_DIRECTORIES}/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(TARGETS ${target}
          EXPORT O2Targets
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endfunction()
