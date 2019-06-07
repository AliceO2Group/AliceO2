include(O2GetTargetDependencies)

#
# o2_get_rpath(target ...) builds a RPATH list using all the dependencies of the
# given target.
#
# * RPATH : the name of the variable that is used to return the rpath values.
#
function(o2_get_rpath)

  message(FATAL_ERROR "review this usage")

  if(BUILD_FOR_DEV)
    return()
  endif()

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        "RPATH"
                        "")
  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(target ${ARGV0})

  set(${A_RPATH}
      $<TARGET_PROPERTY:${target},INTERFACE_LINK_DIRECTORIES>
      PARENT_SCOPE)

endfunction()
