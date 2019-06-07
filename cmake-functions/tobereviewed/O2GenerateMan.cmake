# Generate a man page Make sure we have nroff. If that is not the case we will
# not generate man pages
find_program(NROFF_FOUND nroff)

function(_o2_add_man PARENT NAME)
  if(${PARENT})
    if(NOT TARGET ${PARENT}.man)
      add_custom_target(${PARENT}.man ALL)
      add_dependencies(man ${PARENT}.man)
    endif()
    add_dependencies(${PARENT}.man ${NAME})
  else()
    if(NOT TARGET man)
      add_custom_target(man ALL)
    endif()
    add_dependencies(man ${NAME})
  endif()
endfunction()

function(o2_generate_man)
  cmake_parse_arguments(PARSED_ARGS
                        "" # bool args
                        "NAME;SECTION;MODULE" # mono-valued arguments
                        "" # multi-valued arguments
                        ${ARGN} # arguments
                        )
  if(NOT PARSED_ARGS_SECTION)
    set(PARSED_ARGS_SECTION 1)
  endif()
  if(NOT PARSED_ARGS_NAME)
    message(
      FATAL_ERROR
        "You must provide the name of the input man file in doc/<name>.<section>.in"
      )
  endif()
  if(NROFF_FOUND)
    add_custom_command(
      OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}
      MAIN_DEPENDENCY
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}.in
      COMMAND
        nroff
        -TASCII -MAN
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}.in
        > ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}
      VERBATIM)
    # the prefix man. for the target name avoids circular dependencies for the
    # man pages added at top level. Simply droping the dependency for those does
    # not invoke the custom command on all systems.
    set(CUSTOM_TARGET_NAME man.${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION})

    add_custom_target(
      ${CUSTOM_TARGET_NAME}
      DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION})
    if(PARSED_ARGS_MODULE)
      # add to the man target of specified module
      _o2_add_man(${PARSED_ARGS_MODULE}.man ${CUSTOM_TARGET_NAME})
    elseif(MODULE_NAME)
      # add to the man target of current module
      _o2_add_man(${MODULE_NAME}.man ${CUSTOM_TARGET_NAME})
    else()
      # add to top level target otherwise
      _o2_add_man("" ${CUSTOM_TARGET_NAME})
    endif()
    install(
      FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}
      DESTINATION share/man/man${PARSED_ARGS_SECTION})
  endif(NROFF_FOUND)
endfunction()
