# ######### DEPENDENCIES lookup ############

function(guess_append_libpath _libname _root)
  # Globally adds, as library path, the path of library ${_libname} searched
  # under ${_root}/lib and ${_root}/lib64. The purpose is to work around broken
  # external CMake config files, hardcoding full paths of their dependencies not
  # being relocated properly, leading to broken builds if reusing builds
  # produced under different hosts/paths.
  unset(_lib CACHE) # force find_library to look again
  find_library(_lib "${_libname}"
               HINTS "${_root}" "${_root}/.."
               NO_DEFAULT_PATH
               PATH_SUFFIXES lib lib64)
  if(_lib)
    get_filename_component(_libdir "${_lib}" DIRECTORY)
    message(STATUS "Used to add library path: ${_libdir}")
    link_directories(${_libdir})
  else()
    message(WARNING "Cannot find library ${_libname} under ${_root}")
  endif()
endfunction()

macro(o2_setup_mc)
  if(BUILD_SIMULATION)
    find_package(pythia)
    find_package(Pythia6)

    # Installed via CMake. Note: we work around hardcoded full paths in the
    # CMake config files not being relocated properly by appending library
    # paths.
    
    #guess_append_libpath(geant321 "${Geant3_DIR}")
    find_package(Geant3 MODULE REQUIRED)
    #guess_append_libpath(G4run "${Geant4_DIR}")
    find_package(Geant4 MODULE REQUIRED)
    #guess_append_libpath(geant4vmc "${GEANT4_VMC_DIR}")
    find_package(Geant4VMC MODULE REQUIRED)
    guess_append_libpath(BaseVGM "${VGM_DIR}")

    find_package(VGM NO_MODULE)
    find_package(CERNLIB)
    find_package(HEPMC)

    # check if we have a simulation environment
    if(Geant3_FOUND
       AND Geant4_FOUND
       AND Geant4VMC_FOUND
       AND Pythia6_FOUND
       AND Pythia8_FOUND)
      set(HAVESIMULATION 1)
      message(STATUS "Simulation environment found")
    else()
      message(
        WARNING
          "Simulation environment not found : at least one of the variables Geant3_FOUND , Geant4_FOUND , Geant4VMC_FOUND , Pythia6_FOUND or Pythia8_FOUND is not set"
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
      message(STATUS "Pythia8_FOUND = ${Pythia8_FOUND}")
    endif()
    set(VMCWORKDIR ${CMAKE_INSTALL_PREFIX}/share)
  else()
    message(
      STATUS
        "Simulation not requested : will not look for Geant3, Geant4, Geant4VMC, VGM, CERNLIB, HEPMC"
      )
  endif()
endmacro()
