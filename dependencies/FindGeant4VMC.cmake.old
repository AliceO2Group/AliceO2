# use the GEANT4_VMCConfig.cmake provided by the Geant4VMC installation
# to define Geant4_VMC_XXX varibles we are using below
# to define our target Geant4VMC (note the missing underscore in our
# target name and the difference in case)

find_package(GEANT4VMC NO_MODULE)

message(STATUS "Geant4VMC_INCLUDE_DIRS=${Geant4VMC_INCLUDE_DIRS}")

if(NOT Geant4VMC_INCLUDE_DIRS)
  message(FATAL_ERROR "could not find GEANT4_VMC using config")
endif()

add_library(Geant4VMC SHARED IMPORTED)
set_target_properties(Geant4VMC
                      PROPERTIES IMPORTED_LOCATION
                                 "${LOC}"
                                 INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant4_VMC_INCLUDE_DIRS}")

set(Geant4VMC_FOUND TRUE)
