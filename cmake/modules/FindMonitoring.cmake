# - Tries to find the O2 Monitoring package (include dir and library)
# Author: Barthelemy von Haller
# Author: Adam Wegrzynek
#
#
# This module will set the following non-cached variables:
#  Monitoring_FOUND - states whether Monitoring package has been found
#  Monitoring_INCLUDE_DIRS - Monitoring include directory
#  Monitoring_LIBRARIES - Monitoring library filepath
#  Monitoring_DEFINITIONS - Compiler definitions when comping code using Monitoring
#
# Also following cached variables, but not for general use, are defined:
#  MONITORING_INCLUDE_DIR
#  MONITORING_LIBRARY
#
# This module respects following variables:
#  Monitoring_ROOT - Installation root directory (otherwise it goes through LD_LIBRARY_PATH and ENV)

# Init
include(FindPackageHandleStandardArgs)

# find includes
find_path(MONITORING_INCLUDE_DIR Monitoring.h
           HINTS ${Monitoring_ROOT}/include ENV LD_LIBRARY_PATH PATH_SUFFIXES "../include/Monitoring" "../../include/Monitoring" )

# Remove the final "Monitoring"
get_filename_component(MONITORING_INCLUDE_DIR ${MONITORING_INCLUDE_DIR} DIRECTORY)
set(Monitoring_INCLUDE_DIRS ${MONITORING_INCLUDE_DIR})

# find library
find_library(MONITORING_LIBRARY NAMES Monitoring HINTS ${Monitoring_ROOT}/lib ENV LD_LIBRARY_PATH)
set(Monitoring_LIBRARIES ${MONITORING_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set Monitoring_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Monitoring  "Monitoring could not be found. Set Monitoring_ROOT as root installation directory."
                                  MONITORING_LIBRARY MONITORING_INCLUDE_DIR)
if(${Monitoring_FOUND})
    set(Monitoring_DEFINITIONS "")
    message(STATUS "Monitoring found : ${Monitoring_LIBRARIES}")
endif()

mark_as_advanced(MONITORING_INCLUDE_DIR MONITORING_LIBRARY)
