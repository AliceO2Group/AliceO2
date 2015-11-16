# The name of our project is "ALICEO2".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ALICEO2_SOURCE_DIR}
# or as ${CMAKE_SOURCE_DIR} and to the root binary directory of the project as
# ${ALICEO2_BINARY_DIR} or ${CMAKE_BINARY_DIR}.
# This difference is important for the base classes which are in FAIRROOT
# and the experiment part.

# Check if cmake has the required version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11 FATAL_ERROR)

### CMP0025   Compiler id for Apple Clang is now AppleClang.
### CMP0042   MACOSX_RPATH is enabled by default.

foreach(p
CMP0025 # CMake 3.0
CMP0042 # CMake 3.0
)
if(POLICY ${p})
cmake_policy(SET ${p} NEW)
endif()
endforeach()



# Set name of our project to "ALICEO2". Has to be done
# after check of cmake version since this is a new feature
project(ALICEO2)

#In case you need Fortran
#ENABLE_LANGUAGE(Fortran)

# Check for needed environment variables
#If not stop execution unless modular build is activated
Option(ALICEO2_MODULAR_BUILD "Modular build without environment variables" OFF)
if(NOT ALICEO2_MODULAR_BUILD)
  IF(NOT DEFINED ENV{FAIRROOTPATH})
    MESSAGE(FATAL_ERROR "You did not define the environment variable FAIRROOTPATH which is needed to find FairRoot. Please set this variable and execute cmake again.")
  ENDIF(NOT DEFINED ENV{FAIRROOTPATH})

  IF(NOT DEFINED ENV{SIMPATH})
     MESSAGE(FATAL_ERROR "You did not define the environment variable SIMPATH which is nedded to find the external packages. Please set this variable and execute cmake again.")
  ENDIF(NOT DEFINED ENV{SIMPATH})

  SET(SIMPATH $ENV{SIMPATH})
  SET(FAIRROOTPATH $ENV{FAIRROOTPATH})

  # where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/
  # is checked
  set(CMAKE_MODULE_PATH "${FAIRROOTPATH}/share/fairbase/cmake/modules"  ${CMAKE_MODULE_PATH})
  set(CMAKE_MODULE_PATH "${FAIRROOTPATH}/share/fairbase/cmake/modules_old"  ${CMAKE_MODULE_PATH})
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

  Set(CheckSrcDir "${FAIRROOTPATH}/share/fairbase/cmake/checks")
else(NOT ALICEO2_MODULAR_BUILD)
  find_package(Boost REQUIRED)
endif(NOT ALICEO2_MODULAR_BUILD)

find_package(AliRoot)
find_package(FairRoot)

# Load some basic macros which are needed later on
include(FairMacros)
include(WriteConfigFile)
include(CTest)
include(CheckCompiler)
#include(CheckFortran)

#Check the compiler and set the compile and link flags
If(NOT CMAKE_BUILD_TYPE)
  Message("Set BuildType DEBUG")
  set(CMAKE_BUILD_TYPE Debug)
EndIf(NOT CMAKE_BUILD_TYPE)
Check_Compiler()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(INCLUDE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/include")
Set(VMCWORKDIR ${CMAKE_SOURCE_DIR})
Option(USE_PATH_INFO "Information from PATH and LD_LIBRARY_PATH are used." OFF)
If(USE_PATH_INFO)
  Set(PATH "$PATH")
  If (APPLE)
    Set(LD_LIBRARY_PATH $ENV{DYLD_LIBRARY_PATH})
  Else (APPLE)
    Set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
  EndIf (APPLE)
Else(USE_PATH_INFO)
  STRING(REGEX MATCHALL "[^:]+" PATH $ENV{PATH})
EndIf(USE_PATH_INFO)

# Check if the user wants to build the project in the source
# directory
CHECK_OUT_OF_SOURCE_BUILD()

# Check if we are on an UNIX system. If not stop with an error
# message
IF(NOT UNIX)
  MESSAGE(FATAL_ERROR "You're not on an UNIX system. The project was up to now only tested on UNIX systems, so we break here. If you want to go on please edit the CMakeLists.txt in the source directory.")
ENDIF(NOT UNIX)

# Check if the external packages are installed into a separate install
# directory
CHECK_EXTERNAL_PACKAGE_INSTALL_DIR()

# searches for needed packages
# REQUIRED means that cmake will stop if this packages are not found
# For example the framework can run without GEANT4, but ROOT is
# mandatory

find_package(ROOT 5.32.00 REQUIRED)
find_package(Pythia8)
find_package(Pythia6)
if(ALICEO2_MODULAR_BUILD)
  # Geant3, Geant4 installed via cmake
  find_package(Geant3)
  find_package(Geant4)
else(ALICEO2_MODULAR_BUILD)
  # For old versions of VMC packages (to be removed)
  find_package(GEANT3)
  find_package(GEANT4)
  find_package(GEANT4DATA)
  find_package(GEANT4VMC)
  find_package(CLHEP)
endif(ALICEO2_MODULAR_BUILD)
find_package(CERNLIB)
find_package(HEPMC)
find_package(IWYU)
find_package(DDS)

if(NOT BOOST_ROOT)
  Set(Boost_NO_SYSTEM_PATHS TRUE)
  Set(Boost_NO_BOOST_CMAKE TRUE)
  If(${ROOT_LIBRARY_DIR} MATCHES /lib/root)
    set(BOOST_ROOT ${SIMPATH})
    set(GSL_DIR ${SIMPATH})
  Else(${ROOT_LIBRARY_DIR} MATCHES /lib/root)
    set(BOOST_ROOT ${SIMPATH}/basics/boost)
    set(GSL_DIR ${SIMPATH}/basics/gsl)
  EndIf(${ROOT_LIBRARY_DIR} MATCHES /lib/root)
  Message("-- Looking for Boost ...")
  # If an older version of boost is found both of the variables below are
  # cached and in a second cmake run, a good boost version is found even
  # if the version is to old.
  # To overcome this problem both variables are cleared before checking
  # for boost.
  Unset(Boost_INCLUDE_DIR CACHE)
  Unset(Boost_LIBRARY_DIRS CACHE)
endif(NOT BOOST_ROOT)
find_package(Boost 1.41)
If (Boost_FOUND)
  Set(Boost_Avail 1)
Else (Boost_FOUND)
  Set(Boost_Avail 0)
EndIf (Boost_FOUND)

# set a variable which should be used in all CMakeLists.txt
# Defines all basic include directories from fairbase
if(NOT ALICEO2_MODULAR_BUILD)
  SetBasicVariables()
endif(NOT ALICEO2_MODULAR_BUILD)

# Set the library version in the main CMakeLists.txt
SET(ALICEO2_MAJOR_VERSION 0)
SET(ALICEO2_MINOR_VERSION 0)
SET(ALICEO2_PATCH_VERSION 0)
SET(ALICEO2_VERSION "${ALICEO2_MAJOR_VERSION}.${ALICEO2_MINOR_VERSION}.${ALICEO2_PATCH_VERSION}")
If(NOT ROOT_FOUND_VERSION OR ROOT_FOUND_VERSION LESS 59999)
  SET(FAIRROOT_LIBRARY_PROPERTIES ${FAIRROOT_LIBRARY_PROPERTIES}
  VERSION "${ALICEO2_VERSION}"
  SOVERSION "${ALICEO2_MAJOR_VERSION}"
  SUFFIX ".so"
)
Else()
  SET(FAIRROOT_LIBRARY_PROPERTIES ${FAIRROOT_LIBRARY_PROPERTIES}
  VERSION "${ALICEO2_VERSION}"
  SOVERSION "${ALICEO2_MAJOR_VERSION}"
)
EndIf()

Generate_Version_Info()

SET(_LIBDIR ${CMAKE_BINARY_DIR}/lib)
SET(LD_LIBRARY_PATH  ${_LIBDIR} ${LD_LIBRARY_PATH})

#install(DIRECTORY geometry DESTINATION pnd_install
#          )

if(NOT ALICEO2_MODULAR_BUILD)
# Check if the compiler support specific C++11 features
# Up to now this is only a check since the code does not use
# any of the features of the new standard
# does not work straightforward
include(CheckCXX11Features)

# Check if the compilation flag -std=c++11 is set
If(NOT CMAKE_CXX_FLAGS)
  Message(FATAL_ERROR "No C++11 support found. AliceO2 require C++11 be build.")
EndIF(NOT CMAKE_CXX_FLAGS)

else(NOT ALICEO2_MODULAR_BUILD)
  # This needs to be improved
  # (hopefully by making CheckCXX11Features working also with this build option)
  set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif(NOT ALICEO2_MODULAR_BUILD)

# Recurse into the given subdirectories.  This does not actually
# cause another cmake executable to run.  The same process will walk through
# the project's entire directory structure.

add_subdirectory (Base)
add_subdirectory (Data)
if (PYTHIA8_FOUND AND Pythia6_FOUND)
add_subdirectory (Generators)
Endif ()
add_subdirectory (itsmft)
add_subdirectory (tpc)
add_subdirectory (passive)
add_subdirectory (MathUtils)
add_subdirectory (field)
add_subdirectory (devices)
add_subdirectory (macro)
add_subdirectory (o2cdb)
add_subdirectory (test)


Option(BUILD_DOXYGEN "Build Doxygen" OFF)
if(BUILD_DOXYGEN)
  MESSAGE(STATUS "*** Building the Doxygen documentaion ***")
  ADD_SUBDIRECTORY(doxygen)
endif(BUILD_DOXYGEN)


If(IWYU_FOUND)
  ADD_CUSTOM_TARGET(checkHEADERS 
                    DEPENDS $ENV{ALL_HEADER_RULES}
  )
EndIf()


WRITE_CONFIG_FILE(config.sh)

configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake
               ${CMAKE_BINARY_DIR}/CTestCustom.cmake
              )

# Create an automatically generated config header file to pass file paths to the application
configure_file(${PROJECT_SOURCE_DIR}/header/AliceO2Config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/header/AliceO2Config.h)
