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

include(O2NameTarget)

#
# o2_add_library(baseTargetName SOURCES c1.cxx c2.cxx .....) defines a new
# target of type "library" composed of the given sources. It also defines an
# alias named O2::baseTargetName. The generated library will be called
# libO2[baseTargetName].(dylib|so|.a) (for exact naming see the o2_name_target
# function).
#
# The library will be static or shared depending on the BUILD_SHARED_LIBS option
# (which is normally ON for O2 project)
#
# Parameters:
#
# * SOURCES (required) : the list of source files to compile into this library
#
# * PUBLIC_LINK_LIBRARIES (needed in most cases) : the list of targets this
#   library depends on (e.g. ROOT::Hist, O2::CommonConstants). It is recommended
#   to use the fully qualified target name (i.e. including the namespace part)
#   even for internal (O2) targets.
#
# * PUBLIC_INCLUDE_DIRECTORIES (not needed in most cases) : the list of include
#   directories where to find the include files needed to compile this library
#   and that will be needed as well by the consumers of that library. By default
#   the include subdirectory of the current source directory is taken into
#   account, which should cover most of the use cases. Use this parameter only
#   for special cases then. Note that if you do specify this parameter it
#   replaces the default, it does not add to them.
#
# * PRIVATE_INCLUDE_DIRECTORIES (not needed in most cases) : the list of include
#   directories where to find the include files needed to compile this library,
#   but that will _not_ be needed by its consumers. But default we add the
#   ${CMAKE_CURRENT_BINARY_DIR} here to cover use case of generated headers
#   (e.g. by protobuf). Note that if you do specify this parameter it replaces
#   the default, it does not add to them.
#
function(o2_add_library baseTargetName)

  cmake_parse_arguments(
    PARSE_ARGV
    1
    A
    ""
    "TARGETVARNAME"
    "SOURCES;PUBLIC_INCLUDE_DIRECTORIES;PUBLIC_LINK_LIBRARIES;PRIVATE_INCLUDE_DIRECTORIES"
    )

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  o2_name_target(${baseTargetName} NAME targetName)
  set(target ${targetName})

  # define the target and its O2:: alias
  add_library(${target} ${A_SOURCES})
  add_library(O2::${baseTargetName} ALIAS ${target})

  # set the export name so that packages using O2 can reference the target as
  # O2::${baseTargetName} as well (assuming the export is installed with
  # namespace O2::)
  set_property(TARGET ${target} PROPERTY EXPORT_NAME ${baseTargetName})

  # output name of the lib will be libO2[baseTargetName].(so|dylib|a)
  set_property(TARGET ${target} PROPERTY OUTPUT_NAME O2${baseTargetName})

  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME} ${target} PARENT_SCOPE)
  endif()

  # Start by adding the dependencies to other targets
  if(A_PUBLIC_LINK_LIBRARIES)
    foreach(L IN LISTS A_PUBLIC_LINK_LIBRARIES)
      if(NOT TARGET ${L})
        message(
          FATAL_ERROR "Trying to add a dependency on non-existing target ${L}")
      endif()
      target_link_libraries(${target} PUBLIC ${L})
    endforeach()
  endif()

  # set the public include directories if available
  if(A_PUBLIC_INCLUDE_DIRECTORIES)
    foreach(d IN LISTS A_PUBLIC_INCLUDE_DIRECTORIES)
      get_filename_component(adir ${d} ABSOLUTE)
      if(NOT IS_DIRECTORY ${adir})
        message(
          FATAL_ERROR "Trying to append non existing include directory ${d}")
      endif()
      target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${adir}>)
    endforeach()
  else()
    # use sane default (if it exists)
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include)
      target_include_directories(
        ${target}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
    endif()
  endif()

  # set the private include directories if available
  if(A_PRIVATE_INCLUDE_DIRECTORIES)
    foreach(d IN LISTS A_PRIVATE_INCLUDE_DIRECTORIES)
      get_filename_component(adir ${d} ABSOLUTE)
      if(NOT IS_DIRECTORY ${adir})
        message(
          FATAL_ERROR "Trying to append non existing include directory ${d}")
      endif()
      target_include_directories(${target} PRIVATE $<BUILD_INTERFACE:${d}>)
    endforeach()
  else()
    # use sane(?) default
    target_include_directories(
      ${target}
      PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
    get_filename_component(adir ${CMAKE_CURRENT_LIST_DIR}/src ABSOLUTE)
    if(EXISTS ${adir})
      target_include_directories(
        ${target}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>)
    endif()
  endif()

  if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/include/${baseTargetName})

    # The INCLUDES DESTINATION adds ${CMAKE_INSTALL_INCLUDEDIR} to the
    # INTERFACE_INCLUDE_DIRECTORIES property
    #
    # The EXPORT must come first in the list of parameters
    #
    install(TARGETS ${target}
            EXPORT O2Targets
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

    # install all the includes found in
    # ${CMAKE_CURRENT_LIST_DIR}/include/${baseTargetName} as those are public
    # headers
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/${baseTargetName}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  else()

    # The EXPORT must come first in the list of parameters
    #
    install(TARGETS ${target}
            EXPORT O2Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  endif()

endfunction()
