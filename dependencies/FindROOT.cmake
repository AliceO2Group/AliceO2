# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# small adapter module to fix an issue with ROOTConfig.cmake observed on Mac
# only.
#
# FIXME: to be removed once fixed upstream
#

find_package(${CMAKE_FIND_PACKAGE_NAME}
             ${${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION} NO_MODULE REQUIRED)

if(NOT TARGET vdt)
  find_library(VDT_LIB vdt PATHS ${ROOT_LIBRARY_DIR})
  if(VDT_LIB)
    add_library(vdt SHARED IMPORTED)
    set_target_properties(vdt PROPERTIES IMPORTED_LOCATION ${VDT_LIB})
    message(
      WARNING "vdt target added by hand. Please fix this upstream in ROOT")
  endif()
endif()
