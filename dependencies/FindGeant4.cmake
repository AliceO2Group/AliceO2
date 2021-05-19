# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# use the Geant4Config.cmake provided by the Geant4 installation to create a single target geant4 with the include
# directories and libraries we need

find_package(
  Geant4
  NO_MODULE)
if(NOT
   Geant4_FOUND)
  return()
endif()

add_library(
  geant4
  IMPORTED
  INTERFACE)

set_target_properties(
  geant4
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
             "${Geant4_INCLUDE_DIRS}")

list(
  GET
  Geant4_INCLUDE_DIRS
  0
  Geant4_INCLUDE_DIR)
set(Geant4_LIBRARY_DIRS)
foreach(
  gl4lib
  IN
  LISTS Geant4_LIBRARIES)
  find_library(
    gl4libpath
    NAMES ${gl4lib}
    PATHS "${Geant4_INCLUDE_DIR}/../.."
    PATH_SUFFIXES lib
                  lib64
    NO_DEFAULT_PATH)
  if(gl4libpath)
    get_filename_component(
      gl4libdir
      ${gl4libpath}
      DIRECTORY)
    list(
      APPEND
      Geant4_LIBRARY_DIRS
      ${gl4libdir})
  endif()
  unset(
    gl4libpath
    CACHE)
endforeach()
list(
  REMOVE_DUPLICATES
  Geant4_LIBRARY_DIRS)
set_target_properties(
  geant4
  PROPERTIES INTERFACE_LINK_DIRECTORIES
             "${Geant4_LIBRARY_DIRS}")

# Promote the imported target to global visibility (so we can alias it)
set_target_properties(
  geant4
  PROPERTIES IMPORTED_GLOBAL
             TRUE)

# define a list containing all the variables needed by the physics datasets used by Geant4. The G4ENV list can then be
# used to e.g. define the ENVIRONMENT property of tests that use Geant4
foreach(
  ds
  IN
  LISTS Geant4_DATASETS)
  list(
    APPEND
    G4ENV
    "${Geant4_DATASET_${ds}_ENVVAR}=${Geant4_DATASET_${ds}_PATH}")
endforeach()

add_library(
  MC::Geant4
  ALIAS
  geant4)
