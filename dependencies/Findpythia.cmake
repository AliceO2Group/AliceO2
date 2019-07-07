# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

find_path(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
          NAMES Pythia.h
          PATH_SUFFIXES Pythia8)

find_library(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
             NAMES libpythia8.so libpythia8.dylib)

find_path(${CMAKE_FIND_PACKAGE_NAME}_DATA
          NAMES MainProgramSettings.xml
          PATHS ${${CMAKE_FIND_PACKAGE_NAME}_ROOT}/share/Pythia8/xmldoc)

if(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
   AND ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
   AND ${CMAKE_FIND_PACKAGE_NAME}_DATA)
  add_library(pythia SHARED IMPORTED)
  get_filename_component(incdir ${${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR}/..
                         ABSOLUTE)
  set_target_properties(pythia
                        PROPERTIES IMPORTED_LOCATION
                                   ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED}
                                   INTERFACE_INCLUDE_DIRECTORIES ${incdir})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS ${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
                ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
                ${CMAKE_FIND_PACKAGE_NAME}_DATA)

mark_as_advanced(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
                 ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
                 ${CMAKE_FIND_PACKAGE_NAME}_DATA)
