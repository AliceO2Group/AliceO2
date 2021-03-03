# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# The library provides a Config file for CMake, once installed it can be found via
#
# find_package(Microsoft.GSL CONFIG)
# Which, when successful, will add library target called Microsoft.GSL::GSL which you can use via the usual target_link_libraries mechanism.

find_package(Microsoft.GSL CONFIG)

if(TARGET Microsoft.GSL::GSL)
  # version >= 3.1 now has a proper Config.cmake file
  # so we use that 
  add_library(ms_gsl::ms_gsl ALIAS Microsoft.GSL::GSL)
  set(MS_GSL_FOUND TRUE)
  target_compile_definitions(Microsoft.GSL::GSL INTERFACE MS_GSL_V3)
else()
  
  find_path(MS_GSL_INCLUDE_DIR gsl/gsl PATH_SUFFIXES ms_gsl/include include
          HINTS $ENV{MS_GSL_ROOT})
  
  if(NOT MS_GSL_INCLUDE_DIR)
    set(MS_GSL_FOUND FALSE)
    message(WARNING "MS_GSL not found")
    return()
  endif()
  
  set(MS_GSL_FOUND TRUE)
  
  if(NOT TARGET ms_gsl::ms_gsl)
    add_library(ms_gsl::ms_gsl INTERFACE IMPORTED)
    set_target_properties(ms_gsl::ms_gsl
                          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                     ${MS_GSL_INCLUDE_DIR})
  endif()
  
  mark_as_advanced(MS_GSL_INCLUDE_DIR)
endif()
