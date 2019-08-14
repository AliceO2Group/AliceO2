# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

find_path(MS_GSL_INCLUDE_DIR gsl/gsl PATH_SUFFIXES ms_gsl/include include
        PATHS $ENV{MS_GSL_ROOT})

if(NOT MS_GSL_INCLUDE_DIR)
  set(MS_GSL_FOUND FALSE)
  message(FATAL_ERROR "MS_GSL not found")
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
