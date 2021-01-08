# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(PKGNAME ${CMAKE_FIND_PACKAGE_NAME})
string(TOUPPER ${PKGNAME} PKGENVNAME)

find_library(${PKGNAME}_LIBRARY_SHARED
             NAMES libpythia6.so libpythia6.dylib
             PATHS $ENV{${PKGENVNAME}_ROOT}/lib)

if(${PKGNAME}_LIBRARY_SHARED)
  add_library(pythia6 SHARED IMPORTED)
  set_target_properties(pythia6
                        PROPERTIES IMPORTED_LOCATION
                                   ${${PKGNAME}_LIBRARY_SHARED})
  # Promote the imported target to global visibility (so we can alias it)
  set_target_properties(pythia6 PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(MC::Pythia6 ALIAS pythia6)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
                                  REQUIRED_VARS ${PKGNAME}_LIBRARY_SHARED)

mark_as_advanced(${PKGNAME}_LIBRARY_SHARED)

unset(PKGNAME)
unset(PKGENVNAME)
