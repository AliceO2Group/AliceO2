# small adapter module to fix an issue with ROOTConfig.cmake
# observed on Mac only.
#
# FIXME: to be removed once fixed upstream
#

find_package(${CMAKE_FIND_PACKAGE_NAME} ${${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION} NO_MODULE REQUIRED)

if(APPLE)
  find_library(VDT_LIB vdt PATHS ${ROOT_LIBRARY_DIR})
  if(VDT_LIB)
    add_library(vdt SHARED IMPORTED)
    set_target_properties(vdt PROPERTIES IMPORTED_LOCATION ${VDT_LIB})
    message(
      WARNING "vdt target added by hand. Please fix this upstream in ROOT")
  endif()
endif()
