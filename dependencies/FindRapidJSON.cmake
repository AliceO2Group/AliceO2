# FindRapidJSON.cmake
#
# Finds the rapidjson (header-only) library
#
# This will define the following variables
#
# * RapidJSON_FOUND
#
# and the following imported targets
#
# RapidJSON::RapidJSON
#

find_path(RapidJSON_INC rapidjson.h ${RapidJSON_ROOT}/include/rapidjson)

if(NOT RapidJSON_INC)
  set(RapidJSON_FOUND FALSE)
else()
  set(RapidJSON_FOUND TRUE)
endif()

mark_as_advanced(RapidJSON_INC)

get_filename_component(incdir ${RapidJSON_INC}/.. ABSOLUTE)

if(RapidJSON_FOUND AND NOT TARGET RapidJSON::RapidJSON)
  add_library(RapidJSON::RapidJSON IMPORTED INTERFACE)
  set_target_properties(RapidJSON::RapidJSON
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${incdir})
endif()
