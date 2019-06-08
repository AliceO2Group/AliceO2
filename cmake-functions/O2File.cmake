function(o2_file)

  cmake_parse_arguments(PARSE_ARGV
                        0
                        A
                        ""
                        "DESTINATION"
                        "COPY")
  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  if(IS_ABSOLUTE ${A_DESTINATION})
    message(FATAL_ERROR "DESTINATION should be a relative path")
  endif()

  install(DIRECTORY ${A_COPY}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/${A_DESTINATION})

  file(
    COPY ${A_COPY}
    DESTINATION
      ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR}/${A_DESTINATION})

endfunction()
