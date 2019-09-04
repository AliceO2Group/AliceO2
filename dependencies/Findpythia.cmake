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
          NAMES Pythia.h
          PATH_SUFFIXES Pythia8
          PATHS $ENV{${PKGENVNAME}_ROOT}/include)

find_library(${PKGNAME}_LIBRARY_SHARED
             NAMES libpythia8.so libpythia8.dylib
             PATHS $ENV{${PKGENVNAME}_ROOT}/lib)

find_path(${PKGNAME}_DATA
          NAMES MainProgramSettings.xml
          PATHS ${${PKGNAME}_ROOT}/share/Pythia8/xmldoc
                $ENV{${PKGENVNAME}_ROOT}/share/Pythia8/xmldoc)

if(${PKGNAME}_INCLUDE_DIR AND ${PKGNAME}_LIBRARY_SHARED AND ${PKGNAME}_DATA)
  add_library(pythia SHARED IMPORTED)
  get_filename_component(incdir ${${PKGNAME}_INCLUDE_DIR}/.. ABSOLUTE)
  set_target_properties(pythia
                        PROPERTIES IMPORTED_LOCATION
                                   ${${PKGNAME}_LIBRARY_SHARED}
                                   INTERFACE_INCLUDE_DIRECTORIES ${incdir})
  # Promote the imported target to global visibility (so we can alias it)
  set_target_properties(pythia PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(MC::Pythia ALIAS pythia)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
                                  REQUIRED_VARS ${PKGNAME}_INCLUDE_DIR
                                                ${PKGNAME}_LIBRARY_SHARED
                                                ${PKGNAME}_DATA)

mark_as_advanced(${PKGNAME}_INCLUDE_DIR ${PKGNAME}_LIBRARY_SHARED
                 ${PKGNAME}_DATA)

unset(PKGNAME)
unset(PKGENVNAME)
