include_guard()

#
# o2_add_library(target SOURCES c1.cxx c2.cxx .....) defines a new target
# composed of the given sources. It also defines an alias named O2::target. The
# generated library will be called libO2[target].(dylib|so)
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
#   for special cases then.
#
# * PRIVATE_INCLUDE_DIRECTORIES (not need in most cases) : the list of include
#   directories where to find the include files needed to compile this library,
#   but that will _not_ be needed by its consumers. But default we add the
#   ${CMAKE_CURRENT_BINARY_DIR} here to cover use case of generated headers
#   (e.g. by protobuf).
#
function(o2_add_library)

  cmake_parse_arguments(
    PARSE_ARGV
    1
    A
    ""
    ""
    "SOURCES;PUBLIC_INCLUDE_DIRECTORIES;PUBLIC_LINK_LIBRARIES;PRIVATE_INCLUDE_DIRECTORIES"
    )

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(target ${ARGV0})

  # define the library and its O2:: alias
  add_library(${target} ${A_SOURCES})
  add_library(O2::${target} ALIAS ${target})

  # output name of the lib will be libO2[target]
  set_property(TARGET ${target} PROPERTY OUTPUT_NAME O2${target})

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
      target_include_directories(${target} PUBLIC ${d})
    endforeach()
  else()
    # use sane default (if it exists)
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include)
      target_include_directories(${target}
                                 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
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
      target_include_directories(${target} PRIVATE ${d})
    endforeach()
  else()
    # use sane(?) default
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # will install the library itself
  install(TARGETS ${target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/include/${target})

    # add ${CMAKE_INSTALL_INCLUDEDIR} to the INTERFACE_DIRECTORIES property
    install(TARGETS ${target} INCLUDES ${CMAKE_INSTALL_INCLUDEDIR})

    # install all the includes found in
    # ${CMAKE_CURRENT_LIST_DIR}/include/${target}
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/${target}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  endif()

endfunction()
