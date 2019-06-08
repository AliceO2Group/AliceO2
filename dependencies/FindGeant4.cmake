# use the Geant4Config.cmake provided by the Geant4 installation to create a
# single target geant4 with the include directories and libraries we need

find_package(Geant4 NO_MODULE REQUIRED)

add_library(geant4 IMPORTED INTERFACE)

set_target_properties(geant4
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant4_INCLUDE_DIRS}")
