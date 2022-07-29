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

set(PKGNAME ${CMAKE_FIND_PACKAGE_NAME})
string(TOUPPER ${PKGNAME} PKGENVNAME)

find_path(${PKGNAME}_INCLUDE_DIR
            NAMES wayland-client.h)
find_path(${PKGNAME}_XKBCOMMON_INCLUDE_DIR
            NAMES xkbcommon/xkbcommon.h)
find_library(${PKGNAME}_LIBRARY_SHARED
            NAMES wayland-client libwayland-client)
find_library(${PKGNAME}_XKBCOMMON_LIBRARY_SHARED
            NAMES xkbcommon libxkbcommon)

find_program(${PKGNAME}_SCANNER wayland-scanner)
find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_get_variable(${PKGNAME}_PROTOCOLS_DIR wayland-protocols pkgdatadir)
  if(NOT ${PKGNAME}_PROTOCOLS_DIR OR NOT EXISTS "${${PKGNAME}_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml" OR NOT EXISTS "${${PKGNAME}_PROTOCOLS_DIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml")
    message(STATUS "wayland xdg protocol missing in ${${PKGNAME}_PROTOCOLS_DIR}")
    unset(${PKGNAME}_PROTOCOLS_DIR)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
REQUIRED_VARS ${PKGNAME}_LIBRARY_SHARED ${PKGNAME}_XKBCOMMON_LIBRARY_SHARED ${PKGNAME}_INCLUDE_DIR ${PKGNAME}_XKBCOMMON_INCLUDE_DIR ${PKGNAME}_SCANNER ${PKGNAME}_PROTOCOLS_DIR)

if(${PKGNAME}_LIBRARY_SHARED AND ${PKGNAME}_XKBCOMMON_LIBRARY_SHARED AND ${PKGNAME}_INCLUDE_DIR AND ${PKGNAME}_XKBCOMMON_INCLUDE_DIR AND ${PKGNAME}_SCANNER AND ${PKGNAME}_PROTOCOLS_DIR)
  add_library(WAYLAND-all_libraries INTERFACE)
  target_link_libraries(WAYLAND-all_libraries INTERFACE ${${PKGNAME}_LIBRARY_SHARED} ${${PKGNAME}_XKBCOMMON_LIBRARY_SHARED})
  set_target_properties(WAYLAND-all_libraries PROPERTIES
                                   INTERFACE_INCLUDE_DIRECTORIES ${${PKGNAME}_INCLUDE_DIR}
                                   INTERFACE_INCLUDE_DIRECTORIES ${${PKGNAME}_XKBCOMMON_INCLUDE_DIR})

  add_library(GPUWayland::wayland-client ALIAS WAYLAND-all_libraries)
endif()

mark_as_advanced(${PKGNAME}_LIBRARY_SHARED ${PKGNAME}_INCLUDE_DIR ${PKGNAME}_SCANNER ${PKGNAME}_PROTOCOLS_DIR)

unset(PKGNAME)
unset(PKGENVNAME)
