# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

include_guard()

#
# o2_get_list_of_macros fills ${varname} with the list of macros (*.C files)
# found in directory ${dir}
#
function(o2_get_list_of_macros dir varname)
  file(GLOB_RECURSE listOfMacros RELATIVE ${CMAKE_SOURCE_DIR} ${dir}/*.C)
  # Case sensitive filtering of .C files (to avoid .c files on Mac)
  list(FILTER listOfMacros INCLUDE REGEX "^.*\\.C$")
  # Remove macros that were copied to the build directory, to deal with
  # the (non-recommended-but-can-happen) case where the build directory
  # is a subdirectory of the source dir
  list(FILTER listOfMacros EXCLUDE REGEX "/stage/${CMAKE_INSTALL_DATADIR}")
  set(${varname} ${listOfMacros} PARENT_SCOPE)
endfunction()
