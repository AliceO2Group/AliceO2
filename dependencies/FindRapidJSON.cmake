# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

#
# Finds the rapidjson (header-only) library using the CONFIG file provided by
# RapidJSON and add the RapidJSON::RapidJSON imported targets on top of it
#

find_package(RapidJSON CONFIG QUIET)

if(NOT RapidJSON_INCLUDE_DIR)
  set(RapidJSON_FOUND FALSE)
  if(RapidJSON_FIND_REQUIRED)
    message(FATAL_ERROR "RapidJSON not found")
  endif()
else()
  set(RapidJSON_FOUND TRUE)
endif()

mark_as_advanced(RapidJSON_INCLUDE_DIR)

get_filename_component(inc ${RapidJSON_INCLUDE_DIR} ABSOLUTE)

if(RapidJSON_FOUND AND NOT TARGET RapidJSON::RapidJSON)
  add_library(RapidJSON::RapidJSON IMPORTED INTERFACE)
  set_target_properties(RapidJSON::RapidJSON
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${inc})
endif()

unset(inc)
