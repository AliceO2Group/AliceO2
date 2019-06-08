# use the the config provided by the Geant3 installation but amend the target
# geant321 with the include directories

find_package(Geant3 NO_MODULE REQUIRED)

set_target_properties(geant321
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant3_INCLUDE_DIRS}")
