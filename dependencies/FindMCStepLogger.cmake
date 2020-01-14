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

find_path(${PKGNAME}_INCLUDE_DIR
    NAMES "${PKGNAME}/MCAnalysis.h"
          PATHS $ENV{${PKGENVNAME}_ROOT}/include)

find_library(${PKGNAME}_LIBRARY_SHARED
             NAMES libMCStepLogger.so
             PATHS $ENV{${PKGENVNAME}_ROOT}/lib)

if(${PKGNAME}_INCLUDE_DIR AND ${PKGNAME}_LIBRARY_SHARED)
  add_library(MCStepLogger SHARED IMPORTED)
  get_filename_component(incdir ${${PKGNAME}_INCLUDE_DIR}/.. ABSOLUTE)
  set_target_properties(MCStepLogger
                        PROPERTIES IMPORTED_LOCATION
                                   ${${PKGNAME}_LIBRARY_SHARED}
                                   INTERFACE_INCLUDE_DIRECTORIES ${incdir})
  # Promote the imported target to global visibility (so we can alias it)
  set_target_properties(MCStepLogger PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(MC::MCStepLogger ALIAS MCStepLogger)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
                                  REQUIRED_VARS ${PKGNAME}_INCLUDE_DIR
                                                ${PKGNAME}_LIBRARY_SHARED)

mark_as_advanced(${PKGNAME}_INCLUDE_DIR ${PKGNAME}_LIBRARY_SHARED)

unset(PKGNAME)
unset(PKGENVNAME)
