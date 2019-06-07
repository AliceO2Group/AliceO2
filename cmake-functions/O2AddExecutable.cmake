include_guard()

#
# o2_add_executable(basename SOURCES ...) add an executable with the given
# sources.
#
# * SOURCES (required) gives the list of source files to compile into the
#   executable
# * PUBLIC_LINK_LIBRARIES (needed in most cases) indicates the list of targets
#   this executable depends on.
#
# The installed executable will be named [prefix][-component_name]-exename
#
# where :
#
# * prefix is `o2` unless IS_TEST option is given in which case it is `test`
# * COMPONENT_NAME (optional) is typically used to indicate a subsystem name for
#   regular executables (e.g. o2-tpc-... or o2-mch-...) or the origin target for
#   tests (to help locate the source file in the source directory hierarchy,
#   e.g. test-DataFormats-...)
#
function(o2_add_executable)

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        "IS_TEST;NO_INSTALL;IS_BENCHMARK"
                        "COMPONENT_NAME;EXEVARNAME;TARGETVARNAME"
                        "SOURCES;PUBLIC_LINK_LIBRARIES")

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Got trailing arguments ${A_UNPARSED_ARGUMENTS}")
  endif()

  # set the executable name following our coding convention
  if(A_IS_TEST)
    set(prefix o2-test)
  elseif(A_IS_BENCHMARK)
    set(prefix o2-bench)
  else()
    set(prefix o2)
  endif()

  if(A_COMPONENT_NAME)
    string(TOLOWER ${A_COMPONENT_NAME} component)
    set(prefix ${prefix}-${component})
  endif()

  set(exeName ${prefix}-${ARGV0})

  # register the computed names into output variables.
  if(A_EXEVARNAME)
    get_filename_component(exeFullPath ${exeName} REALPATH BASE_DIR
                           ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(${A_EXEVARNAME} ${exeFullPath} PARENT_SCOPE)
  endif()
  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME} ${exeName} PARENT_SCOPE)
  endif()

  # add the executable with its sources
  add_executable(${exeName} ${A_SOURCES})

  # use its dependencies
  foreach(lib IN LISTS A_PUBLIC_LINK_LIBRARIES)
    if(NOT TARGET ${lib})
      message(
        FATAL_ERROR "Trying to add a dependency on non-existing target ${lib}")
    endif()
    target_link_libraries(${exeName} PUBLIC ${lib})
  endforeach()

  if(NOT A_NO_INSTALL)
    # install the executable

    if(A_IS_TEST)
      install(TARGETS ${exeName}
              RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)
    else()
      install(TARGETS ${exeName} RUNTIME)
    endif()
  endif()

endfunction()
