# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

function(o2_define_output_paths)

  # Set CMAKE_INSTALL_LIBDIR explicitly to lib (to avoid lib64 on CC7)
  set(CMAKE_INSTALL_LIBDIR lib PARENT_SCOPE)

  include(GNUInstallDirs)

  if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_BINDIR}
        PARENT_SCOPE)
  endif()
  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_LIBDIR}
        PARENT_SCOPE)
  endif()
  if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_LIBDIR}
        PARENT_SCOPE)
  endif()

endfunction()
