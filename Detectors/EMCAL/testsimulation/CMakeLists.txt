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

if(NOT BUILD_TEST_ROOT_MACROS)
  return()
endif()

set(genmacros fixedEnergyElectronGun.C fixedEnergyPionGun.C fixedEnergyPhotonGun.C fixedEnergyMuonGun.C)

foreach(genmacro ${genmacros})
  MESSAGE(STATUS "Adding test for macro ${genmacro}")
  o2_add_test_root_macro(${genmacro}
    COMPILE_ONLY
    PUBLIC_LINK_LIBRARIES FairRoot::Base
    LABELS emcal)
  o2_data_file(COPY ${genmacro} DESTINATION Detectors/EMC/testsimulation)
endforeach()

o2_add_test_root_macro(PutEmcalInTop.C
  PUBLIC_LINK_LIBRARIES O2::EMCALSimulation
  O2::DetectorsPassive FairRoot::Base
  LABELS emcal)

o2_add_test_root_macro(drawEMCALgeometry.C
  PUBLIC_LINK_LIBRARIES O2::EMCALSimulation
  O2::DetectorsPassive FairRoot::Base
  LABELS emcal)

o2_add_test_root_macro(run_sim_emcal.C
  PUBLIC_LINK_LIBRARIES FairRoot::Base O2::EMCALSimulation
  O2::DetectorsPassive O2::Generators
  LABELS emcal)
