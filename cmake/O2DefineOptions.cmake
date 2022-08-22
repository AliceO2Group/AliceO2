# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

include_guard()

function(o2_define_options)

  option(BUILD_SHARED_LIBS "Build shared libs" ON)

  option(BUILD_ANALYSIS "Build analysis parts" ON)

  option(BUILD_EXAMPLES "Build examples" ON)

  option(BUILD_TEST_ROOT_MACROS
         "Build the tests toload and compile the Root macros" ON)

  option(ENABLE_CASSERT "Enable asserts" OFF)

  option(
    BUILD_SIMULATION_DEFAULT
    "Default behavior for simulation (disregarded if BUILD_SIMULATION is defined)"
    ON)
  # for the complete picture of how BUILD_SIMULATION is handled see
  # ../dependencies/O2SimulationDependencies.cmake

  option(ENABLE_UPGRADES "Enable detectors for upgrades" OFF)

  option(ENABLE_THREAD_SAFETY_ANALYSIS "Enable thread safety analysis" OFF)
endfunction()
