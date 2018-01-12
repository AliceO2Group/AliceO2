# The name of our project is "ALICEO2".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ALICEO2_SOURCE_DIR}
# or as ${CMAKE_SOURCE_DIR} and to the root binary directory of the project as
# ${ALICEO2_BINARY_DIR} or ${CMAKE_BINARY_DIR}.
# This difference is important for the base classes which are in FAIRROOT
# and the experiment part.

# Check IF cmake has the required version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11 FATAL_ERROR)

### CMP0025   Compiler id for Apple Clang is now AppleClang.
### CMP0042   MACOSX_RPATH is enabled by default.

FOREACH (p
    CMP0025 # CMake 3.0
    CMP0042 # CMake 3.0
    )
  IF (POLICY ${p})
    cmake_policy(SET ${p} NEW)
  ENDIF ()
endforeach ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# Set name of our project to "ALICEO2". Has to be done
# after check of cmake version since this is a new feature
project(ALICEO2)

# Toplevel targets
ADD_CUSTOM_TARGET(man ALL)
#In case you need Fortran
#ENABLE_LANGUAGE(Fortran)

# Check for needed environment variables
#IF not stop execution unless modular build is activated
Option(ALICEO2_MODULAR_BUILD "Modular build without environment variables" OFF)
IF (NOT ALICEO2_MODULAR_BUILD)
  IF (NOT DEFINED ENV{FAIRROOTPATH})
    MESSAGE(FATAL_ERROR "You did not define the environment variable FAIRROOTPATH which is needed to find FairRoot. Please set this variable and execute cmake again.")
  ENDIF (NOT DEFINED ENV{FAIRROOTPATH})

  IF (NOT DEFINED ENV{SIMPATH})
    MESSAGE(FATAL_ERROR "You did not define the environment variable SIMPATH which is nedded to find the external packages. Please set this variable and execute cmake again.")
  ENDIF (NOT DEFINED ENV{SIMPATH})

  SET(SIMPATH $ENV{SIMPATH})
  SET(FAIRROOTPATH $ENV{FAIRROOTPATH})

  # where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/
  # is checked
  set(CMAKE_MODULE_PATH "${FAIRROOTPATH}/share/fairbase/cmake/modules" ${CMAKE_MODULE_PATH})
  set(CMAKE_MODULE_PATH "${FAIRROOTPATH}/share/fairbase/cmake/modules_old" ${CMAKE_MODULE_PATH})
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

  Set(CheckSrcDir "${FAIRROOTPATH}/share/fairbase/cmake/checks")
ELSE (NOT ALICEO2_MODULAR_BUILD)
  find_package(Boost REQUIRED)
ENDIF (NOT ALICEO2_MODULAR_BUILD)

# Load some basic macros which are needed later on
include(O2Utils)
include(O2Dependencies)
include(FairMacros)
include(WriteConfigFile)
include(CTest)
include(CheckCompiler)
#include(CheckFortran)

# check if we have a simulation environment
# (unfortunately neither GEANT4_FOUND nor GEANT3_FOUND work reliably)
find_library(G4RUNLIB libG4run.dylib libG4run.so HINTS ENV LD_LIBRARY_PATH)
find_library(G3RUNLIB libgeant321.dylib libgeant321.so HINTS ENV LD_LIBRARY_PATH)
find_library(VMCLIB libgeant4vmc.dylib libgeant4vmc.so HINTS ENV LD_LIBRARY_PATH)
if (G4RUNLIB AND G3RUNLIB AND VMCLIB)
  SET (HAVESIMULATION 1)
  message(STATUS "Simulation environment found")
endif()

# Build type for coverage builds
set(CMAKE_CXX_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_COVERAGE}")
set(CMAKE_Fortran_FLAGS_COVERAGE "-g -O2 -fprofile-arcs -ftest-coverage")
set(CMAKE_LINK_FLAGS_COVERAGE "--coverage -fprofile-arcs  -fPIC")

MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_Fortran_FLAGS_COVERAGE
    CMAKE_LINK_FLAGS_COVERAGE)


#Check the compiler and set the compile and link flags
IF (NOT CMAKE_BUILD_TYPE)
  Message(STATUS "Set BuildType DEBUG")
  set(CMAKE_BUILD_TYPE Debug)
ENDIF (NOT CMAKE_BUILD_TYPE)
Check_Compiler()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(INCLUDE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/include")
Set(VMCWORKDIR ${CMAKE_SOURCE_DIR})
Option(USE_PATH_INFO "Information from PATH and LD_LIBRARY_PATH are used." OFF)
IF (USE_PATH_INFO)
  Set(PATH "$PATH")
  IF (APPLE)
    Set(LD_LIBRARY_PATH $ENV{DYLD_LIBRARY_PATH})
  ELSE (APPLE)
    Set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
  ENDIF (APPLE)
ELSE (USE_PATH_INFO)
  STRING(REGEX MATCHALL "[^:]+" PATH "$ENV{PATH}")
ENDIF (USE_PATH_INFO)

# Check IF the user wants to build the project in the source
# directory
CHECK_OUT_OF_SOURCE_BUILD()

# Check IF we are on an UNIX system. IF not stop with an error
# message
IF (NOT UNIX)
  MESSAGE(FATAL_ERROR "You're not on an UNIX system. The project was up to now only tested on UNIX systems, so we break here. IF you want to go on please edit the CMakeLists.txt in the source directory.")
ENDIF (NOT UNIX)

# Check IF the external packages are installed into a separate install
# directory
CHECK_EXTERNAL_PACKAGE_INSTALL_DIR()

# set a variable which should be used in all CMakeLists.txt
# Defines all basic include directories from fairbase
IF (NOT ALICEO2_MODULAR_BUILD)
  SetBasicVariables()
ENDIF (NOT ALICEO2_MODULAR_BUILD)

# Set the library version in the main CMakeLists.txt
SET(ALICEO2_MAJOR_VERSION 0)
SET(ALICEO2_MINOR_VERSION 0)
SET(ALICEO2_PATCH_VERSION 0)
SET(ALICEO2_VERSION "${ALICEO2_MAJOR_VERSION}.${ALICEO2_MINOR_VERSION}.${ALICEO2_PATCH_VERSION}")
IF (NOT ROOT_FOUND_VERSION OR ROOT_FOUND_VERSION LESS 59999)
  SET(FAIRROOT_LIBRARY_PROPERTIES ${FAIRROOT_LIBRARY_PROPERTIES}
      VERSION "${ALICEO2_VERSION}"
      SOVERSION "${ALICEO2_MAJOR_VERSION}"
      SUFFIX ".so"
      )
ELSE ()
  SET(FAIRROOT_LIBRARY_PROPERTIES ${FAIRROOT_LIBRARY_PROPERTIES}
      VERSION "${ALICEO2_VERSION}"
      SOVERSION "${ALICEO2_MAJOR_VERSION}"
      )
ENDIF ()

Generate_Version_Info()

SET(_LIBDIR ${CMAKE_BINARY_DIR}/lib)
SET(LD_LIBRARY_PATH ${_LIBDIR} ${LD_LIBRARY_PATH})

IF (NOT ALICEO2_MODULAR_BUILD)
  # Check IF the compiler support specific C++14 features
  # Up to now this is only a check since the code does not use
  # any of the features of the new standard
  # does not work straightforward
  include(CheckCXX11Features)

  # Check IF the compilation flag -std=c++14 is set
  IF (NOT CMAKE_CXX_FLAGS)
    Message(FATAL_ERROR "No C++14 support found. AliceO2 requires C++14 be build.")
  EndIF (NOT CMAKE_CXX_FLAGS)

ELSE (NOT ALICEO2_MODULAR_BUILD)
  # This needs to be improved
  # (hopefully by making CheckCXX11Features working also with this build option)
  set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
ENDIF (NOT ALICEO2_MODULAR_BUILD)

#check C++14 features
set(CheckCXX14SrcDir "${CMAKE_SOURCE_DIR}/cmake/checks")
include(CheckCXX14Features)

# Recurse into the given subdirectories.  This does not actually
# cause another cmake executable to run.  The same process will walk through
# the project's entire directory structure.

add_subdirectory(Generators)
set(GENERATORS_LIBRARY Generators)

add_subdirectory(CCDB)
add_subdirectory(Common)
add_subdirectory(DataFormats)
add_subdirectory(Detectors)
add_subdirectory(EventVisualisation)
add_subdirectory(Examples)
add_subdirectory(Framework)
add_subdirectory(Algorithm)
add_subdirectory(macro)
add_subdirectory(Utilities)
add_subdirectory(Steer)
add_subdirectory(doc)
if (HAVESIMULATION)
   add_subdirectory(run)
endif()
add_subdirectory(config)

IF (IWYU_FOUND)
  ADD_CUSTOM_TARGET(checkHEADERS
      DEPENDS $ENV{ALL_HEADER_RULES}
      )
ENDIF ()

Install(FILES ${CMAKE_BINARY_DIR}/check_system.sh
        DESTINATION bin
       )

SET(VMCWORKDIR ${CMAKE_SOURCE_DIR})

WRITE_CONFIG_FILE(config.sh)

SET(VMCWORKDIR ${CMAKE_INSTALL_PREFIX}/share)
SET(ROOT_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)
WRITE_CONFIG_FILE(config.sh_install)

Install(FILES ${CMAKE_BINARY_DIR}/config.sh_install
        DESTINATION bin
        RENAME O2Config.sh
       )

# Place the CTestCustom.cmake in the build dir
configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake
    ${CMAKE_BINARY_DIR}/CTestCustom.cmake
    )

O2_GENERATE_MAN(NAME o2)
O2_GENERATE_MAN(NAME FairMQDevice)

# Run macros unit tests automatically if ALIBUILD_O2_TESTS is defined
IF($ENV{ALIBUILD_O2_TESTS})
  SET(CHECK_ROOTMACRO_COMPILE 1)
ENDIF()

# UNIT TESTS VERIFYING CONSISTENT STATE OF OUR ROOT MACROS AND THE EXECUTION ENVIRONMENT
if(CHECK_ROOTMACRO_COMPILE AND HAVESIMULATION)
  FIND_PROGRAM(BASH_PROGRAM bash)

  # On Mac OS GLOB_RECURSE returns both .C and .c files, i.e. case insensitive
  FILE(GLOB_RECURSE MACRO_FILES "*.C")
  # Case sensitive filtering of .C files
  LIST(FILTER MACRO_FILES INCLUDE REGEX "^.*\\.C$")
  SET(TEST_NO 0)
  FOREACH(MACRO_FILE ${MACRO_FILES})
    MATH(EXPR TEST_NO "${TEST_NO} + 1")
    SET(EXE_NAME ${CMAKE_BINARY_DIR}/tmp_scripts/macro_test${TEST_NO}.sh)

    # Create test script for macros
    FILE(WRITE ${EXE_NAME} "export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib\n root -b -l -q -e \".L ${MACRO_FILE}\"")

    ADD_TEST(NAME ${MACRO_FILE}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMAND ${BASH_PROGRAM} ${EXE_NAME}
    )
  ENDFOREACH()
endif()
