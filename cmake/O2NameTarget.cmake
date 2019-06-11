include_guard()

#
# o2_name_target(baseName NAME var ...) gives a project specific name to the
# target of the given baseName. The computed name is retrieved in the variable
# "var".
#
# * NAME var: will contain the computed name of the target
# * IS_TEST: present to denote the target is a test executable
# * IS_BENCH: present to denote the target is a benchmark executable
# * IS_EXE: present to denote the target is an executable (and not a library)
#
function(o2_name_target)

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        "IS_TEST;IS_BENCH;IS_EXE"
                        "NAME;COMPONENT_NAME"
                        "")

  set(baseTargetName ${ARGV0})

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT A_NAME)
    message(FATAL_ERROR "Parameter NAME is mandatory")
  endif()

  # get the target "type" (lib or exe,test,bench)
  if(A_IS_TEST)
    set(targetType test)
  elseif(A_IS_BENCHMARK)
    set(targetType bench)
  elseif(A_IS_EXE)
    set(targetType exe)
  else()
    set(targetType lib)
  endif()

  # get the component part if present
  if(A_COMPONENT_NAME)
    string(TOLOWER ${A_COMPONENT_NAME} component)
    set(comp -${component})
  endif()

  set(${A_NAME} ${PROJECT_NAME}${targetType}${comp}-${baseTargetName} PARENT_SCOPE)

endfunction()
