################################################################################
# Module for locating XRootD.
#
#   XROOTD_FOUND
#     Indicates whether the library has been found.
#
#   XROOTD_INCLUDE_DIRS
#      Specifies XRootD include directory.
#
#   XROOTD_LIBRARIES
#     Specifies XRootD libraries that should be passed to target_link_libararies.
#
#   XROOTD_<COMPONENT>_LIBRARIES
#     Specifies the libraries of a specific <COMPONENT>
#
#   XROOTD_<COMPONENT>_FOUND
#     Indicates whether the specified <COMPONENT> was found.
#
#   List of components: CLIENT, UTILS, SERVER, POSIX, HTTP and SSI
################################################################################

# ######################################################################################################################
# Set XRootD include paths
# ######################################################################################################################
find_path(
  XROOTD_INCLUDE_DIRS
  XrdVersion.hh
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES include/xrootd
  PATHS /opt/xrootd)

if(NOT
   "${XROOTD_INCLUDE_DIRS}"
   STREQUAL
   "XROOTD_INCLUDE_DIRS-NOTFOUND")
  set(XROOTD_FOUND TRUE)
endif()

if(NOT
   XROOTD_FOUND)
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD client libs - libXrdCl
# ######################################################################################################################
find_library(
  XROOTD_CLIENT_LIBRARIES
  XrdCl
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_CLIENT_LIBRARIES}"
   STREQUAL
   "XROOTD_CLIENT_LIBRARIES-NOTFOUND")
  set(XROOTD_CLIENT_FOUND TRUE)
  list(
    APPEND
    XROOTD_LIBRARIES
    ${XROOTD_CLIENT_LIBRARIES})
endif()

if(XRootD_FIND_REQUIRED_CLIENT
   AND NOT
       XROOTD_CLIENT_FOUND)
  message("XRootD client required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_CLIENT_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD utils libs - libXrdUtils
# ######################################################################################################################
find_library(
  XROOTD_UTILS_LIBRARIES
  XrdUtils
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_UTILS_LIBRARIES}"
   STREQUAL
   "XROOTD_UTILS_LIBRARIES-NOTFOUND")
  set(XROOTD_UTILS_FOUND TRUE)
  list(
    APPEND
    XROOTD_LIBRARIES
    ${XROOTD_UTILS_LIBRARIES})
endif()

if(XRootD_FIND_REQUIRED_UTILS
   AND NOT
       XROOTD_UTILS_FOUND)
  message("XRootD utils required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_UTILS_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD server libs - libXrdServer
# ######################################################################################################################
find_library(
  XROOTD_SERVER_LIBRARIES
  XrdServer
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_SERVER_LIBRARIES}"
   STREQUAL
   "XROOTD_SERVER_LIBRARIES-NOTFOUND")
  set(XROOTD_SERVER_FOUND TRUE)
  list(
    APPEND
    XROOTD_LIBRARIES
    ${XROOTD_SERVER_LIBRARIES})
endif()

if(XRootD_FIND_REQUIRED_SERVER
   AND NOT
       XROOTD_SERVER_FOUND)
  message("XRootD server required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_SERVER_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD posix libs - libXrdPosix - libXrdPosixPreload
# ######################################################################################################################
find_library(
  XROOTD_POSIX_LIBRARY
  XrdPosix
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

find_library(
  XROOTD_POSIX_PRELOAD_LIBRARY
  XrdPosixPreload
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_POSIX_LIBRARY}"
   STREQUAL
   "XROOTD_POSIX_LIBRARY-NOTFOUND")
  if(NOT
     "${XROOTD_POSIX_PRELOAD_LIBRARY}"
     STREQUAL
     "XROOTD_POSIX_PRELOAD_LIBRARY-NOTFOUND")
    set(XROOTD_POSIX_LIBRARIES
        ${XROOTD_POSIX_LIBRARY}
        ${XROOTD_POSIX_PRELOAD_LIBRARY})
    set(XROOTD_POSIX_FOUND TRUE)
    list(
      APPEND
      XROOTD_LIBRARIES
      ${XROOTD_POSIX_LIBRARIES})
  endif()
endif()

if(XRootD_FIND_REQUIRED_POSIX
   AND NOT
       XROOTD_POSIX_FOUND)
  message("XRootD posix required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_POSIX_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD HTTP (XrdHttp) libs - libXrdHtppUtils
# ######################################################################################################################
find_library(
  XROOTD_HTTP_LIBRARIES
  XrdHttpUtils
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_HTTP_LIBRARIES}"
   STREQUAL
   "XROOTD_HTTP_LIBRARIES-NOTFOUND")
  set(XROOTD_HTTP_FOUND TRUE)
  list(
    APPEND
    XROOTD_LIBRARIES
    ${XROOTD_HTTP_LIBRARIES})
endif()

if(XRootD_FIND_REQUIRED_HTTP
   AND NOT
       XROOTD_HTTP_FOUND)
  message("XRootD http required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_HTTP_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# XRootD SSI libs - XrdSsiLib - XrdSsiShMap
# ######################################################################################################################
find_library(
  XROOTD_SSI_LIBRARY
  XrdSsiLib
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

find_library(
  XROOTD_SSI_SHMAP_LIBRARY
  XrdSsiShMap
  HINTS ${XROOTD_DIR}
        $ENV{XROOTD_DIR}
        /usr
        /opt/xrootd
  PATH_SUFFIXES lib
                lib64)

if(NOT
   "${XROOTD_SSI_LIBRARY}"
   STREQUAL
   "XROOTD_SSI_LIBRARY-NOTFOUND")
  if(NOT
     "${XROOTD_SSI_SHMAP_LIBRARY}"
     STREQUAL
     "XROOTD_SSI_SHMAP_LIBRARY-NOTFOUND")
    set(XROOTD_SSI_LIBRARIES
        ${XROOTD_SSI_LIBRARY}
        ${XROOTD_SSI_SHMAP_LIBRARY})
    set(XROOTD_SSI_FOUND TRUE)
    list(
      APPEND
      XROOTD_LIBRARIES
      ${XROOTD_SSI_LIBRARIES})
  endif()
endif()

if(XRootD_FIND_REQUIRED_SSI
   AND NOT
       XROOTD_SSI_FOUND)
  message("XRootD ssi required but not found!")
  list(
    APPEND
    _XROOTD_MISSING_COMPONENTS
    XROOTD_SSI_FOUND)
  unset(XROOTD_FOUND)
endif()

# ######################################################################################################################
# Set up the XRootD find module
# ######################################################################################################################

if(XRootD_FIND_REQUIRED)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    XRootD
    REQUIRED_VARS XROOTD_INCLUDE_DIRS
                  ${_XROOTD_MISSING_COMPONENTS})
endif()

if(XROOTD_CLIENT_FOUND)
  if(NOT
     TARGET
     XRootD::Client)
    get_filename_component(
      libdir
      ${XROOTD_CLIENT_LIBRARIES}
      DIRECTORY)
    add_library(
      XRootD::Client
      INTERFACE
      IMPORTED)
    set_target_properties(
      XRootD::Client
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                 ${XROOTD_INCLUDE_DIRS}
                 INTERFACE_LINK_LIBRARIES
                 ${XROOTD_CLIENT_LIBRARIES}
                 INTERFACE_LINK_DIRECTORIES
                 ${libdir})
  endif()
endif()
