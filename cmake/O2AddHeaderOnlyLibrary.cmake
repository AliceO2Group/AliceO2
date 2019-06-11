include_guard()

# o2_add_header_only_library creates a header-only target.
#
# * INCLUDE_DIRECTORIES the relative path(s) to the headers of that library if
#   not specified will be set as "include" simply (which should work just fine
#   in most cases)
#
function(o2_add_header_only_library)

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

  set(target ${ARGV0})

  add_library(${target} INTERFACE)
  add_library(O2::${target} ALIAS ${target})

  if(NOT A_INCLUDE_DIRECTORIES)
    get_filename_component(dir include ABSOLUTE)
    if(EXISTS ${dir})
      set(A_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${dir}>)
    else()
      set(A_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
    endif()
  endif()

  target_include_directories(${target} INTERFACE ${A_INCLUDE_DIRECTORIES})

  if(A_INTERFACE_LINK_LIBRARIES)
    target_link_libraries(${target} INTERFACE ${A_INTERFACE_LINK_LIBRARIES})
  endif()
  install(DIRECTORY ${A_INCLUDE_DIRECTORIES}/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endfunction()
