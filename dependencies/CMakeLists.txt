include_guard()

include("${CMAKE_CURRENT_LIST_DIR}/O2RecipeAdapter.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

if(ALIBUILD_BASEDIR)
  # try autodetecting external packages from an alibuild installation zone
  include(O2FindDependenciesFromAliBuild)
  o2_find_dependencies_from_alibuild(${ALIBUILD_BASEDIR} LABEL ${ALIBUILD_LABEL}
                                     QUIET)
endif()

# Required packages
#
# Order is not completely irrelevant. For instance arrow must come before
# FairRoot (see FindFairRoot.cmake)
#
# Generally speaking we should prefer the CONFIG variant of the find_package. We
# explicitely don't use the CONFIG variant (i.e. we do use the MODULE variant)
# only for some packages XXX where we define our own FindXXX.cmake module (e.g.
# to complement and/or fix what's done in the package's XXXConfig.cmake file)

include(FindThreads)

find_package(arrow CONFIG REQUIRED)
find_package(Vc REQUIRED)
find_package(ROOT 6.06.00 MODULE REQUIRED)
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
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
find_package(InfoLogger REQUIRED)
find_package(Configuration REQUIRED)
find_package(Monitoring REQUIRED)
find_package(Common REQUIRED)
find_package(RapidJSON MODULE REQUIRED)
find_package(CURL REQUIRED)

# MC specific packages
if(BUILD_SIMULATION)
  find_package(pythia MODULE REQUIRED)
  find_package(pythia6 MODULE REQUIRED)
  find_package(Geant3 MODULE REQUIRED)
  find_package(Geant4 MODULE REQUIRED)
  find_package(Geant4VMC MODULE REQUIRED)
  find_package(VGM CONFIG REQUIRED)
  find_package(HepMC CONFIG REQUIRED)
endif()

# Optional packages

find_package(DDS)
find_package(benchmark CONFIG)
find_package(OpenMP)
find_package(GLFW NAMES glfw3 CONFIG)
find_package(AliRoot)
