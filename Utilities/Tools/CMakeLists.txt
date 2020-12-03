# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -print-file-name=libc.so.6
                 OUTPUT_VARIABLE GLIBC
                 OUTPUT_STRIP_TRAILING_WHITESPACE)

GET_FILENAME_COMPONENT (GLIBC ${GLIBC} REALPATH)
GET_FILENAME_COMPONENT (GLIBC_VERSION ${GLIBC} NAME)
STRING (REPLACE "libc-" "" GLIBC_VERSION ${GLIBC_VERSION})
STRING (REPLACE ".so" "" GLIBC_VERSION ${GLIBC_VERSION})
if(${GLIBC_VERSION} VERSION_LESS 2.32)
  add_subdirectory(cpulimit) # Incompatible to glibc >= 2.32
endif()

install(PROGRAMS monitor-mem.sh DESTINATION share/scripts/)
install(PROGRAMS jobutils.sh DESTINATION share/scripts/)
install(PROGRAMS grid_submit.sh DESTINATION share/scripts/)
