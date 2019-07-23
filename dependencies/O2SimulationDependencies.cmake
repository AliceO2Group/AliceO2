# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

#
# Note that the BUILD_SIMULATION option governs what to do with the simulation
# parts of the repository, depending on whether or not the MC simulation
# packages (pythia, geant, etc...) needed for those parts are available or not.
#
# If BUILD_SIMULATION is specified on the command line (using -D), or set in the
# cache, then it is a hard requirement :
#
# * BUILD_SIMULATION=ON and MCpackages found => BUILD_SIMULATION=ON
# * BUILD_SIMULATION=ON and MCpackages not found => FAILURE
# * BUILD_SIMULATION=OFF => BUILD_SIMULATION=OFF (regardless of MCpackages found
#   or not)
#
# If on the other hand BUILD_SIMULATION is NOT specified on the command line
# then simulation is built if MCpackages are available and the default value for
# build simulation is set to ON
#
# * MCpackages found => BUILD_SIMULATION=BUILD_SIMULATION_DEFAULT
# * MCpackages not found => BUILD_SIMULATION=OFF
#

include_guard()

set(mcPackageRequirement OPTIONAL)
if(DEFINED BUILD_SIMULATION AND BUILD_SIMULATION)
  set(mcPackageRequirement REQUIRED)
endif()

# MC specific packages
find_package(pythia MODULE)
set_package_properties(pythia
                       PROPERTIES
                       TYPE ${mcPackageRequirement} DESCRIPTION
                            "the Pythia8 generator")
find_package(pythia6 MODULE)
set_package_properties(pythia6
                       PROPERTIES
                       TYPE ${mcPackageRequirement} DESCRIPTION
                            "the Pythia6 legacy generator")
find_package(Geant3 MODULE)
set_package_properties(Geant3
                       PROPERTIES
                       TYPE ${mcPackageRequirement} DESCRIPTION
                            "the legacy but not slow MC transport engine")
find_package(Geant4 MODULE)
set_package_properties(Geant4
                       PROPERTIES
                       TYPE ${mcPackageRequirement} DESCRIPTION
                            "more recent and more complete MC transport engine")
find_package(Geant4VMC MODULE)
set_package_properties(Geant4VMC PROPERTIES TYPE ${mcPackageRequirement})
find_package(VGM CONFIG)
set_package_properties(VGM PROPERTIES TYPE ${mcPackageRequirement})
find_package(HepMC CONFIG)
set_package_properties(HepMC PROPERTIES TYPE ${mcPackageRequirement})

set(doBuildSimulation OFF)

if(pythia_FOUND
   AND pythia6_FOUND
   AND Geant3_FOUND
   AND Geant4_FOUND
   AND Geant4VMC_FOUND
   AND VGM_FOUND
   AND HepMC_FOUND)
  set(doBuildSimulation ON)
endif()

if(DEFINED BUILD_SIMULATION AND BUILD_SIMULATION AND NOT doBuildSimulation)
  return()
endif()

if(NOT DEFINED BUILD_SIMULATION)
  if(NOT BUILD_SIMULATION_DEFAULT)
    option(BUILD_SIMULATION "Build simulation related parts" FALSE)
  else()
    option(BUILD_SIMULATION "Build simulation related parts"
           ${doBuildSimulation})
  endif()
endif()
