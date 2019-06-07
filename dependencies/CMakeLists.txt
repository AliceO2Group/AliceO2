include_guard()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

if(ALIBUILD_BASEDIR)
  # try autodetecting external packages from an alibuild installation zone
  include(O2FindDependenciesFromAliBuild)
  o2_find_dependencies_from_alibuild(${ALIBUILD_BASEDIR} LABEL ${ALIBUILD_LABEL}
                                     QUIET)
endif()

# Required packages 
#
# Order is not completely irrelevant.
# For instance arrow must come before FairRoot (see FindFairRoot.cmake)

include(FindThreads)

find_package(arrow CONFIG REQUIRED)
find_package(Vc REQUIRED)
find_package(ROOT 6.06.00 NO_MODULE REQUIRED)
find_package(Boost 1.59
             COMPONENTS container
                        thread
                        system
                        timer
                        program_options
                        random
                        filesystem
                        chrono
                        exception
                        regex
                        serialization
                        log
                        log_setup
                        unit_test_framework
                        date_time
                        signals
                        iostreams
             REQUIRED)
find_package(FairLogger CONFIG REQUIRED)
find_package(FairRoot MODULE REQUIRED)
find_package(ms_gsl MODULE REQUIRED)
find_package(FairMQ REQUIRED)
find_package(DDS)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
find_package(InfoLogger REQUIRED)
find_package(Configuration REQUIRED)
find_package(Monitoring REQUIRED)
find_package(Common REQUIRED)
find_package(RapidJSON MODULE REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(CURL REQUIRED)

# Optional packages

find_package(OpenMP)
find_package(GLFW NAMES GLFW3 CONFIG)
find_package(AliRoot)

# MC specific packages

if(BUILD_SIMULATION)
  find_package(pythia)
  find_package(Pythia6)

  find_package(Geant3 MODULE REQUIRED)
  find_package(Geant4 MODULE REQUIRED)
  find_package(Geant4VMC MODULE REQUIRED)
  find_package(VGM NO_MODULE)
  find_package(CERNLIB)
  find_package(HEPMC)

  # check if we have a simulation environment
  if(Geant3_FOUND
     AND Geant4_FOUND
     AND Geant4VMC_FOUND
     AND Pythia6_FOUND
     AND pythia_FOUND)
    set(HAVESIMULATION 1)
    message(STATUS "Simulation environment found")
  else()
    message(
      WARNING
        "Simulation environment not found : at least one of the variables Geant3_FOUND , Geant4_FOUND , Geant4VMC_FOUND , Pythia6_FOUND or pythia_FOUND is not set"
      )
    message(WARNING "All of them are needed for a simulation environment.")
    message(
      WARNING
        "That might not be a problem if you don't care about simulation though."
      )
    message(STATUS "Geant3_FOUND = ${Geant3_FOUND}")
    message(STATUS "Geant4_FOUND = ${Geant4_FOUND}")
    message(STATUS "Geant4VMC_FOUND = ${Geant4VMC_FOUND}")
    message(STATUS "Pythia6_FOUND = ${Pythia6_FOUND}")
    message(STATUS "pythia_FOUND = ${pythia_FOUND}")
  endif()
endif()

include(O2CUDA)

# FIXME : those DDS-related definitions should not be global ones but be tied to
# some targets (or even not at all if DDS targets are well defined ?) *
# if(DDS_FOUND) *   add_definitions(-DENABLE_DDS) * add_definitions(-DDDS_FOUND)
# *   set(OPTIONAL_DDS_LIBRARIES ${DDS_INTERCOM_LIBRARY_SHARED} *
# ${DDS_PROTOCOL_LIBRARY_SHARED} * ${DDS_USER_DEFAULTS_LIBRARY_SHARED}) *
# set(OPTIONAL_DDS_INCLUDE_DIR ${DDS_INCLUDE_DIR}) * endif()
