# - Try to find the O2 Configuration package include dirs and libraries
# Author: Barthelemy von Haller
#
# This script will set the following variables:
#  Configuration_FOUND - System has Configuration
#  Configuration_INCLUDE_DIRS - The Configuration include directories
#  Configuration_LIBRARIES - The libraries needed to use Configuration
#  Configuration_DEFINITIONS - Compiler switches required for using Configuration
#
# This script can use the following variables:
#  Configuration_ROOT - Installation root to tell this module where to look. (it tries LD_LIBRARY_PATH otherwise)

# Init
include(FindPackageHandleStandardArgs)

# find includes
find_path(CONFIGURATION_INCLUDE_DIR ConfigurationInterface.h
           HINTS ${Configuration_ROOT}/include ENV LD_LIBRARY_PATH PATH_SUFFIXES "../include/Configuration" "../../include/Configuration" )
# Remove the final "Configuration"
get_filename_component(CONFIGURATION_INCLUDE_DIR ${CONFIGURATION_INCLUDE_DIR} DIRECTORY)
set(Configuration_INCLUDE_DIRS ${CONFIGURATION_INCLUDE_DIR})

# find library
find_library(CONFIGURATION_LIBRARY NAMES Configuration HINTS ${Configuration_ROOT}/lib ENV LD_LIBRARY_PATH)
set(Configuration_LIBRARIES ${CONFIGURATION_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set CONFIGURATION_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Configuration  "Configuration could not be found. Install package Configuration or set Configuration_ROOT to its root installation directory."
                                  CONFIGURATION_LIBRARY CONFIGURATION_INCLUDE_DIR)

if(${CONFIGURATION_FOUND})
    message(STATUS "Configuration found : ${Configuration_LIBRARIES}")
endif()

mark_as_advanced(CONFIGURATION_INCLUDE_DIR CONFIGURATION_LIBRARY)
