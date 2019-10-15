# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(genmacros fixedEnergyElectronGun.C fixedEnergyPionGun.C fixedEnergyPhotonGun.C)
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
