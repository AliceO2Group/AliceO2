# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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

  foreach(D IN LISTS A_COPY)
    get_filename_component(adir ${D} ABSOLUTE)
    if(IS_DIRECTORY ${adir})
      install(DIRECTORY ${D}
              DESTINATION ${CMAKE_INSTALL_DATADIR}/${A_DESTINATION})
    else()

      install(FILES ${D} DESTINATION ${CMAKE_INSTALL_DATADIR}/${A_DESTINATION})
    endif()
  endforeach()

  file(
    COPY ${A_COPY}
    DESTINATION
      ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR}/${A_DESTINATION})

endfunction()
