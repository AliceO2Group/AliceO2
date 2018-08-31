# - Try to find the O2 Common package include dirs and libraries
# Author: Barthelemy von Haller
#
# This script will set the following variables:
#  Common_FOUND - System has Common
#  Common_INCLUDE_DIRS - The Common include directories
#  Common_LIBRARIES - The libraries needed to use Common
#  Common_DEFINITIONS - Compiler switches required for using Common
#
# This script can use the following variables:
#  Common_ROOT - Installation root to tell this module where to look. (it tries LD_LIBRARY_PATH otherwise)

# Init
include(FindPackageHandleStandardArgs)

# find includes
find_path(COMMON_INCLUDE_DIR Timer.h
           HINTS ${Common_ROOT}/include ENV LD_LIBRARY_PATH PATH_SUFFIXES "../include/Common" "../../include/Common" )
# Remove the final "Common"
get_filename_component(COMMON_INCLUDE_DIR ${COMMON_INCLUDE_DIR} DIRECTORY)
set(Common_INCLUDE_DIRS ${COMMON_INCLUDE_DIR})

# find library
find_library(COMMON_LIBRARY NAMES Common HINTS ${Common_ROOT}/lib ENV LD_LIBRARY_PATH)
set(Common_LIBRARIES ${COMMON_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set COMMON_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Common  "Common could not be found. Install package Common or set Common_ROOT to its root installation directory."
                                  COMMON_LIBRARY COMMON_INCLUDE_DIR)

if(${COMMON_FOUND})
    message(STATUS "Common found : ${Common_LIBRARIES}")
endif()

mark_as_advanced(COMMON_INCLUDE_DIR COMMON_LIBRARY)
