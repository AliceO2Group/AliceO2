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
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND TRUE)
  if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
    message(
      STATUS
        "pythia library found: ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED}")
    message(
      STATUS "pythia includes found: ${${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR}")
    message(STATUS "pythia data found: ${${CMAKE_FIND_PACKAGE_NAME}_DATA}")
  endif()
  add_library(pythia SHARED IMPORTED)
  set_target_properties(pythia
                        PROPERTIES IMPORTED_LOCATION
                                   ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED}
                                   INTERFACE_INCLUDE_DIRECTORIES
                                   ${${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR}/..)
else()
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
      "Sorry. Could not find pythia8")
endif()

mark_as_advanced(${CMAKE_FIND_PACKAGE_NAME}_INCLUDE_DIR
                 ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
                 ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_DATA)
