# Here the idea is to reuse the Geant4Config.cmake that is provided by the
# Geant4 project, to define our own Geant4 target (whereas the Geant4 project
# defines a geant321 target, for one single configuration which might not be the
# one we want/need)
#

find_package(Geant4 NO_MODULE)

if(NOT Geant4_FOUND)
  message(FATAL_ERROR "could not find Geant4 using config")
endif()

message(STATUS "Geant4_LIBRARIES=${Geant4_LIBRARIES}")
message(STATUS "Geant4_INCLUDE_DIRS=${Geant4_INCLUDE_DIRS}")

add_library(Geant4 SHARED IMPORTED)
set_target_properties(Geant4
                      PROPERTIES INTERFACE_LINK_LIBRARIES
                                 "${Geant4_LIBRARIES}"
                                 INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant4_INCLUDE_DIRS}")

set(Geant4_FOUND TRUE)
