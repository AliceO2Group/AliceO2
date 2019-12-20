# Find module for libjalienO2.
# Variables:
#  - LIBJALIENO2_FOUND: TRUE if found
#  - LIBJALIENO2_LIBPATH: library path
#  - LIBJALIENO2_INCLUDE_DIR: include dir

# Init
include(FindPackageHandleStandardArgs)

# find includes
find_path(LIBJALIENO2_INCLUDE_DIR TJAlienSSLContext.h
  PATH_SUFFIXES "include"
  HINTS "${LIBJALIENO2}"
)

mark_as_advanced(LIBJALIENO2_INCLUDE_DIR)

# find library
find_library(LIBJALIENO2_LIBPATH "jalienO2"
  PATH_SUFFIXES "lib"
  HINTS "${LIBJALIENO2}"
)

mark_as_advanced(LIBJALIENO2_LIBPATH)

find_package_handle_standard_args(libjalienO2 DEFAULT_MSG
                                  LIBJALIENO2_LIBPATH LIBJALIENO2_INCLUDE_DIR)

if(LIBJALIENO2_FOUND)
  set(LIBJALIENO2_LIBRARIES ${LIBJALIENO2_LIBPATH})
  set(LIBJALIENO2_INCLUDE_DIRS ${LIBJALIENO2_INCLUDE_DIR})

  # add target
  if(NOT TARGET libjalien::libjalienO2)
    add_library(libjalien::libjalienO2 IMPORTED INTERFACE)
    set_target_properties(libjalien::libjalienO2 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LIBJALIENO2_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${LIBJALIENO2_LIBPATH}"
    )
  endif()
endif()
