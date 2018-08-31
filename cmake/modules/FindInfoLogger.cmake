# - Tries to find the O2 InfoLogger package (include dir and library)
# Author: Barthelemy von Haller
# Author: Adam Wegrzynek
# Author: Sylvain Chapeland
#
# This module will set the following non-cached variables:
#  InfoLogger_FOUND - states whether InfoLogger package has been found
#  InfoLogger_INCLUDE_DIRS - InfoLogger include directory
#  InfoLogger_LIBRARIES - InfoLogger library filepath
#  InfoLogger_DEFINITIONS - Compiler definitions when comping code using InfoLogger
#
# Also following cached variables, but not for general use, are defined:
#  INFOLOGGER_INCLUDE_DIR
#  INFOLOGGER_LIBRARY
#
# This module respects following variables:
#  InfoLogger_ROOT - Installation root directory (otherwise it goes through LD_LIBRARY_PATH and ENV)

# Init
include(FindPackageHandleStandardArgs)

# Need Common
find_package(Common REQUIRED)

# find includes
find_path(INFOLOGGER_INCLUDE_DIR InfoLogger.hxx
           HINTS ${InfoLogger_ROOT}/include ENV LD_LIBRARY_PATH PATH_SUFFIXES "../include/InfoLogger" "../../include/InfoLogger" )

# Remove the final "InfoLogger"
get_filename_component(INFOLOGGER_INCLUDE_DIR ${INFOLOGGER_INCLUDE_DIR} DIRECTORY)
set(InfoLogger_INCLUDE_DIRS ${INFOLOGGER_INCLUDE_DIR})

# find library
find_library(INFOLOGGER_LIBRARY NAMES InfoLogger HINTS ${InfoLogger_ROOT}/lib ENV LD_LIBRARY_PATH)
set(InfoLogger_LIBRARIES ${INFOLOGGER_LIBRARY} ${Common_LIBRARIES})

# handle the QUIETLY and REQUIRED arguments and set InfoLogger_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(InfoLogger  "InfoLogger could not be found. Set InfoLogger_ROOT as root installation directory."
                                  INFOLOGGER_LIBRARY INFOLOGGER_INCLUDE_DIR)
if(${InfoLogger_FOUND})
    set(InfoLogger_DEFINITIONS "")
    message(STATUS "InfoLogger found : ${InfoLogger_LIBRARIES}")
endif()

mark_as_advanced(INFOLOGGER_INCLUDE_DIR INFOLOGGER_LIBRARY)
