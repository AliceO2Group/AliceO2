# use the GEANT4_VMCConfig.cmake provided by the Geant4VMC installation but
# amend the target geant4vmc with the include directories

find_package(Geant4VMC NO_MODULE REQUIRED)

set_target_properties(geant4vmc
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant4VMC_INCLUDE_DIRS}")
