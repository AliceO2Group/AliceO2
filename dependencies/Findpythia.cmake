find_path(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
          NAMES Pythia.h
          PATH_SUFFIXES Pythia8)

find_library(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
             NAMES libpythia8.so libpythia8.dylib)
# PATHS ${${CMAKE_FIND_PACKAGE_NAME}_ROOT}/lib)

find_path(${CMAKE_FIND_PACKAGE_NAME}_DATA
          NAMES MainProgramSettings.xml
          PATHS ${${CMAKE_FIND_PACKAGE_NAME}_ROOT}/share/Pythia8/xmldoc)

if(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
   AND ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
   AND ${CMAKE_FIND_PACKAGE_NAME}_DATA)
  add_library(pythia SHARED IMPORTED)
  set_target_properties(pythia
                        PROPERTIES IMPORTED_LOCATION
                                   ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED}
                                   INTERFACE_INCLUDE_DIRECTORIES
                                   ${${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR}/..)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS ${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
                ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
                ${CMAKE_FIND_PACKAGE_NAME}_DATA)

mark_as_advanced(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
                 ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
                 ${CMAKE_FIND_PACKAGE_NAME}_DATA)
