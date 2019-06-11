#
# o2_data_file(COPY src DESTINATION dest) is a convenience function to copy and
# install src into dest in a single command. dest should be a relative path.
#
# The install occurs only in the installation phase (if any) and puts src into
# ${CMAKE_INSTALL_DATADIR}/dest
#
# The copy always happens at configure time and puts src into
# ${CMAKE_BINARY_DIR}/stage/{CMAKE_INSTALL_DATADIR}/dest
#
# Note that when src denotes directories src and src/ means different things :
#
# o2_add_file(COPY src/ DESTINATION dest) will copy the _content_ of src into
# dest, while o2_add_file(COPY src DESTINATION dest) will copy the directory src
# into dest.
#
function(o2_data_file)

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
