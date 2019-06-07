# Here the idea is to reuse the Geant3Config.cmake that is provided by the
# Geant3 project, to define our own Geant3 target (whereas the Geant3 project
# defines a geant321 target, for one single configuration which might not be the
# one we want/need)
#

find_package(Geant3 NO_MODULE)

if(NOT Geant3_FOUND)
  message(FATAL_ERROR "could not find Geant3 using config")
endif()

# FIXME: should be more general here and handle the case where Geant3 is built
# using (an)other configuration(s), not just RELWITHDEBINFO

# get_target_property(LIL geant321
# IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO)
get_target_property(LOC geant321 IMPORTED_LOCATION_RELWITHDEBINFO)

add_library(Geant3 SHARED IMPORTED)
set_target_properties(Geant3
                      PROPERTIES IMPORTED_LOCATION ${LOC}
                                 INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant3_INCLUDE_DIRS}")

set(Geant3_FOUND TRUE)
