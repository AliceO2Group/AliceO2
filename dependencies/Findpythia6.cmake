
find_library(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED
             NAMES libpythia6.so libpythia6.dylib)

if(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)
  add_library(pythia6 SHARED IMPORTED)
  set_target_properties(pythia6
                        PROPERTIES IMPORTED_LOCATION
                                   ${${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS ${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)

mark_as_advanced(${CMAKE_FIND_PACKAGE_NAME}_LIBRARY_SHARED)
