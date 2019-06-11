# The name of our project is "ALICEO2".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ALICEO2_SOURCE_DIR}
# or as ${CMAKE_SOURCE_DIR} and to the root binary directory of the project as
# ${ALICEO2_BINARY_DIR} or ${CMAKE_BINARY_DIR}.
# This difference is important for the base classes which are in FAIRROOT
# and the experiment part.

# Check IF cmake has the required version
CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0 FATAL_ERROR)

### CMP0025   Compiler id for Apple Clang is now AppleClang.
### CMP0042   MACOSX_RPATH is enabled by default.

FOREACH (p
    CMP0025 # CMake 3.0
    CMP0042 # CMake 3.0
    CMP0028
    CMP0068
    CMP0057
    )
  IF (POLICY ${p})
    cmake_policy(SET ${p} NEW)
  ENDIF ()
endforeach ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# Set name of our project to "ALICEO2". Has to be done
# after check of cmake version since this is a new feature
project(ALICEO2)

# Toplevel targets
ADD_CUSTOM_TARGET(man ALL)
#In case you need Fortran
#ENABLE_LANGUAGE(Fortran)

find_package(Boost REQUIRED)

# Load some basic macros which are needed later on
include(O2Utils)
include(O2Dependencies)
include(FairMacros)
include(WriteConfigFile)
include(CTest)
#include(CheckFortran)

# check if we have a simulation environment
if (Geant3_FOUND AND Geant4_FOUND AND Geant4VMC_FOUND AND Pythia6_FOUND AND PYTHIA8_FOUND)
  SET (HAVESIMULATION 1)
  message(STATUS "Simulation environment found")
else()
  message(STATUS "Simulation environment not found : at least one of the variables Geant3_FOUND , Geant4_FOUND , Geant4VMC_FOUND , Pythia6_FOUND or PYTHIA8_FOUND is not set")
  message(STATUS "All of them are needed for a simulation environment.")
  message(STATUS "That might not be a problem if you don't care about simulation though.")
  message(STATUS "Geant3_FOUND    = ${Geant3_FOUND}")
  message(STATUS "Geant4_FOUND    = ${Geant4_FOUND}")
  message(STATUS "Geant4VMC_FOUND = ${Geant4VMC_FOUND}")
  message(STATUS "Pythia6_FOUND=${Pythia6_FOUND}")
  message(STATUS "PYTHIA8_FOUND=${PYTHIA8_FOUND}")
  message(FATAL "stop")
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

IF(ENABLE_CASSERT) #For the CI, we want to have <cassert> assertions enabled
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
ELSE()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
ENDIF()
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_Fortran_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g")
# make sure Debug build not optimized (does not seem to work without CACHE + FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "Debug mode build flags" FORCE)
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug mode build flags" FORCE)

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE} - CXXFLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

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

# Our libraries will be under "lib"
SET(_LIBDIR ${CMAKE_BINARY_DIR}/lib)
SET(LD_LIBRARY_PATH ${_LIBDIR} ${LD_LIBRARY_PATH})

# Build targets with install rpath on Mac to dramatically speed up installation
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
  endif()
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()
unset(isSystemDir)

# Check for the required C++ standard features (break if not available)
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
SET(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples")
if (BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()
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
add_subdirectory(GPU)

IF (IWYU_FOUND)
  ADD_CUSTOM_TARGET(checkHEADERS
      DEPENDS $ENV{ALL_HEADER_RULES}
      )
ENDIF ()

SET(VMCWORKDIR ${CMAKE_SOURCE_DIR})

SET(VMCWORKDIR ${CMAKE_INSTALL_PREFIX}/share)
SET(ROOT_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)

# Place the CTestCustom.cmake in the build dir
configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake
    ${CMAKE_BINARY_DIR}/CTestCustom.cmake
    )

O2_GENERATE_MAN(NAME o2)
O2_GENERATE_MAN(NAME FairMQDevice)

# Macros from this list will be excluded from tests. To be used only in exceptional cases, such as
# when the macro uses symbols from outside the standard O2 build/runtime environment
set(EXCLUDE_MACROS_FROM_TEST

  ${CMAKE_SOURCE_DIR}/Generators/share/external/hijing.C
  ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/macros/EVE/rootlogon.C

  # Temporarily disable the following macros: ROOT v6-14-04 fails to parse
  # correctly some Boost headers
  ${CMAKE_SOURCE_DIR}/CCDB/example/fill_local_ocdb.C
  ${CMAKE_SOURCE_DIR}/macro/loadExtDepLib.C
  ${CMAKE_SOURCE_DIR}/macro/putCondition.C

  # Exclude AliRoot macros from AliGPU package
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger/macros/checkPropagation.C
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger/macros/fitPolynomialFieldIts.C
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger/macros/fitPolynomialFieldTpc.C
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger/macros/fitPolynomialFieldTrd.C
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Standalone/tools/dump.C
  ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TRDTracking/macros/checkDbgOutput.C
  ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/initTPCcalibration.C
  ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/createTPCFastTransform.C
  ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/moveTPCFastTransform.C
  ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/loadlibs.C

  # This macro is used by the o2-sim tests, no need to re-check. Parallel tests
  # of the same macro are also breaking it
  ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C
)

# Macros from this list will have the test in compiled mode ran in a non-fatal way
set(EXCLUDE_MACROS_FROM_COMPILED_TEST
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/calibration/macro/comparePedestalsAndNoise.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/calibration/macro/drawNoiseAndPedestal.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/monitor/macro/RunCompareMode3.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/monitor/macro/RunFindAdcError.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/macro/dEdxRes.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/macro/readClusters.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/macro/testTracks.C
  ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/macro/readMCtruth.C
  ${CMAKE_SOURCE_DIR}/macro/convertClusterToClusterHardware.C
  ${CMAKE_SOURCE_DIR}/macro/run_clus_tpc.C
)

# ROOT on MacOS has linking issues with Vc, disabled until fixed
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(EXCLUDE_MACROS_FROM_COMPILED_TEST
    ${EXCLUDE_MACROS_FROM_COMPILED_TEST}
    ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/IrregularSpline1DTest.C
    ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation/macro/IrregularSpline2D3DTest.C
  )
endif()

# UNIT TESTS VERIFYING CONSISTENT STATE OF OUR ROOT MACROS AND THE EXECUTION ENVIRONMENT
if(HAVESIMULATION)
  # On Mac OS GLOB_RECURSE returns both .C and .c files, i.e. case insensitive
  file(GLOB_RECURSE MACRO_FILES "*.C")
  # Case sensitive filtering of .C files
  list(FILTER MACRO_FILES INCLUDE REGEX "^.*\\.C$")
  foreach(MACRO_FILE ${MACRO_FILES})
    if(NOT ${MACRO_FILE} IN_LIST EXCLUDE_MACROS_FROM_TEST)
      string(REPLACE ${CMAKE_SOURCE_DIR} "" MACRO_FILE_LABEL ${MACRO_FILE})

      # Test for "interpreted" macros
      add_test_wrap(NAME ${MACRO_FILE_LABEL}
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND root -n -b -l -q -e ".L ${MACRO_FILE}")

      # Test for compiled macros
      if(NOT ${MACRO_FILE} IN_LIST EXCLUDE_MACROS_FROM_COMPILED_TEST)
        add_test_wrap(NAME ${MACRO_FILE_LABEL}_compiled
                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                      ${_MACRO_NON_FATAL}
                      COMMAND root -n -b -l -q -e ".L ${MACRO_FILE}++")

        # Set environment variables
        foreach(_TEST_NAME "${MACRO_FILE_LABEL}" "${MACRO_FILE_LABEL}_compiled")
          set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")
          if(APPLE)
            set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT "DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")
          endif()
          set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT "ROOT_HIST=0")
        endforeach()

        # Cleanup
        unset(_TEST_NAME)
      endif()
    endif()
  endforeach()
endif()

# Create special .rootrc for testing compiled macros
configure_file("${CMAKE_SOURCE_DIR}/cmake/tests.rootrc.in"
               "${CMAKE_BINARY_DIR}/.rootrc"
               @ONLY
               NEWLINE_STYLE UNIX)

# Create tests wrapper (and make it executable)
configure_file("${CMAKE_SOURCE_DIR}/cmake/tests-wrapper.sh.in"
               "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/tests-wrapper.sh"
               @ONLY
               NEWLINE_STYLE UNIX)
file(COPY "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/tests-wrapper.sh"
    DESTINATION "${CMAKE_BINARY_DIR}"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Create test for executable naming convention
configure_file("${CMAKE_SOURCE_DIR}/cmake/ensure-executable-naming-convention.sh.in"
               "${CMAKE_BINARY_DIR}/ensure-executable-naming-convention.sh"
               @ONLY
               NEWLINE_STYLE UNIX)

add_test(NAME "ensure-executable-naming-convention"
        COMMAND "ensure-executable-naming-convention.sh")
