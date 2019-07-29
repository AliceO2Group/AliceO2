# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

find_library(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
             NAMES libpythia6.so libpythia6.dylib)

if(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)
  add_library(pythia6 SHARED IMPORTED)
  set_target_properties(pythia6
                        PROPERTIES IMPORTED_LOCATION
                                   ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)

mark_as_advanced(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)
