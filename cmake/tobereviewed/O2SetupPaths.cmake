macro(o2_setup_paths)

  set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
  set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
  set(INCLUDE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/include")
  set(VMCWORKDIR ${CMAKE_SOURCE_DIR})
  option(USE_PATH_INFO "Information from PATH and LD_LIBRARY_PATH are used."
         OFF)
  if(USE_PATH_INFO)
    set(PATH "$PATH")
    if(APPLE)
      set(LD_LIBRARY_PATH $ENV{DYLD_LIBRARY_PATH})
    else(APPLE)
      set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
    endif(APPLE)
  else(USE_PATH_INFO)
    string(REGEX MATCHALL "[^:]+" PATH "$ENV{PATH}")
  endif(USE_PATH_INFO)

  # Our libraries will be under "lib"
  set(_LIBDIR ${CMAKE_BINARY_DIR}/lib)
  set(LD_LIBRARY_PATH ${_LIBDIR} ${LD_LIBRARY_PATH})

  # Build targets with install rpath on Mac to dramatically speed up
  # installation
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
            "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
    endif()
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
  unset(isSystemDir)

endmacro()
