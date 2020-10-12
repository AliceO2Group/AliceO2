# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Author: Jochen Klein
#
# add FastJet::FastJet as library to targets depending on FastJet,
# place code depending on FastJet in blocks:
# if(FastJet_FOUND) ... endif() (in cmake)
# #ifdef HAVE_FASTJET ... #endif (in C++)

set(PKGNAME ${CMAKE_FIND_PACKAGE_NAME})
string(TOUPPER ${PKGNAME} PKGENVNAME)
string(TOLOWER "${PKGNAME}-config" PKGCONFIG)

find_program(${PKGNAME}_CONFIG
             NAMES ${PKGCONFIG}
             PATHS $ENV{${PKGENVNAME}})
mark_as_advanced(${PKGNAME}_CONFIG)

if(${PKGNAME}_CONFIG)
  execute_process(COMMAND ${${PKGNAME}_CONFIG} --cxxflags
                  OUTPUT_VARIABLE ${PKGNAME}_CXXFLAGS
                  ERROR_VARIABLE error
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(${PKGNAME}_INCLUDE_DIRS)
  mark_as_advanced(${PKGNAME}_INCLUDE_DIRS)
  if(${PKGNAME}_CXXFLAGS)
    string(REGEX MATCHALL "(^| )-I[^ ]+" incdirs ${${PKGNAME}_CXXFLAGS})
    foreach(incdir ${incdirs})
      string(STRIP ${incdir} incdir)
      string(SUBSTRING ${incdir} 2 -1 incdir)
      list(APPEND ${PKGNAME}_INCLUDE_DIRS ${incdir})
    endforeach()
  endif(${PKGNAME}_CXXFLAGS)

  execute_process(COMMAND ${${PKGNAME}_CONFIG} --libs
                  OUTPUT_VARIABLE ${PKGNAME}_CONFIGLIBS
                  ERROR_VARIABLE error
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(${PKGNAME}_LIBS)
  set(${PKGNAME}_LIB_DIRS)
  mark_as_advanced(${PKGNAME}_LIBS ${PKGNAME}_LIB_DIRS)
  if(${PKGNAME}_CONFIGLIBS)
    string(REGEX MATCHALL "(^| )-l[^ ]+" libs ${${PKGNAME}_CONFIGLIBS})
    foreach(lib ${libs})
        string(STRIP ${lib} lib)
        string(SUBSTRING ${lib} 2 -1 lib)
        list(APPEND ${PKGNAME}_LIBS ${lib})
    endforeach()

    string(REGEX MATCHALL "(^| )-L[^ ]+" libdirs ${${PKGNAME}_CONFIGLIBS})
    foreach(libdir ${libdirs})
        string(STRIP ${libdir} libdir)
        string(SUBSTRING ${libdir} 2 -1 libdir)
        list(APPEND ${PKGNAME}_LIB_DIRS ${libdir})
    endforeach()

    # append directories of exposed libraries
    if ("CGAL" IN_LIST ${PKGNAME}_LIBS)
      find_library(lib_cgal NAMES "CGAL" PATHS $ENV{CGAL_ROOT}/lib NO_DEFAULT_PATH)
      if (NOT lib_cgal)
        message(FATAL_ERROR "CGAL not found in $ENV{CGAL_ROOT}/lib")
      endif()
      get_filename_component(dir_cgal ${lib_cgal} DIRECTORY)
      list(APPEND ${PKGNAME}_LIB_DIRS ${dir_cgal})
    endif()

    if ("gmp" IN_LIST ${PKGNAME}_LIBS)
      find_library(lib_gmp NAMES "gmp" PATHS $ENV{GMP_ROOT}/lib NO_DEFAULT_PATH)
      if (NOT lib_gmp)
        message(FATAL_ERROR "GMP not found in $ENV{GMP_ROOT}/lib")
      endif()
      get_filename_component(dir_gmp ${lib_gmp} DIRECTORY)
      list(APPEND ${PKGNAME}_LIB_DIRS ${dir_gmp})
    endif()
  endif(${PKGNAME}_CONFIGLIBS)
endif(${PKGNAME}_CONFIG)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKGNAME}
                                  REQUIRED_VARS ${PKGNAME}_INCLUDE_DIRS
                                                ${PKGNAME}_LIB_DIRS
                                                ${PKGNAME}_LIBS)

# if everything was found, assemble target with dependencies
if(${${PKGNAME}_FOUND})
  add_library(${PKGNAME}::${PKGNAME} IMPORTED INTERFACE GLOBAL)
  target_compile_definitions(${PKGNAME}::${PKGNAME} INTERFACE "HAVE_${PKGENVNAME}")

  foreach(lib ${${PKGNAME}_LIBS})
    target_link_libraries(${PKGNAME}::${PKGNAME} INTERFACE ${lib})
  endforeach()

  foreach(libdir ${${PKGNAME}_LIB_DIRS})
    target_link_directories(${PKGNAME}::${PKGNAME} INTERFACE ${libdir})
  endforeach()

  foreach(incdir ${${PKGNAME}_INCLUDE_DIRS})
    target_include_directories(${PKGNAME}::${PKGNAME} INTERFACE ${incdir})
  endforeach()

  find_library(lib_contrib NAMES "fastjetcontribfragile" PATHS ${${PKGNAME}_LIB_DIRS} NO_DEFAULT_PATH)
  if(lib_contrib)
    message(STATUS "adding FastJet contrib")
    add_library(${PKGNAME}::Contrib IMPORTED INTERFACE GLOBAL)
    target_link_libraries(${PKGNAME}::Contrib INTERFACE ${lib_contrib})
  endif()
endif()

unset(PKGNAME)
unset(PKGENVNAME)
