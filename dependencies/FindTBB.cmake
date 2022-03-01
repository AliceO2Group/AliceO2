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

find_package(${CMAKE_FIND_PACKAGE_NAME} NO_MODULE)

if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FOUND)

set(PKGNAME ${CMAKE_FIND_PACKAGE_NAME})
string(TOUPPER ${PKGNAME} PKGENVNAME)

find_library(${PKGNAME}_LIBRARY_SHARED
             NAMES libtbb.so libtbb.dylib)

if(${PKGNAME}_LIBRARY_SHARED)
  add_library(tbb SHARED IMPORTED)
  set_target_properties(tbb
                        PROPERTIES IMPORTED_LOCATION
                                   ${${PKGNAME}_LIBRARY_SHARED})
  # Promote the imported target to global visibility (so we can alias it)
  set_target_properties(tbb PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(TBB::tbb ALIAS tbb)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
                                  REQUIRED_VARS ${PKGNAME}_LIBRARY_SHARED)

mark_as_advanced(${PKGNAME}_LIBRARY_SHARED)

unset(PKGNAME)
unset(PKGENVNAME)

endif()
