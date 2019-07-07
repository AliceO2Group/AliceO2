# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_test(DigitContainer
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCSimulation
            COMPONENT_NAME tpc
            SOURCES testTPCDigitContainer.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(ElectronTransport
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCSimulation
            COMPONENT_NAME tpc
            SOURCES testTPCElectronTransport.cxx)

o2_add_test(GEMAmplification
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCSimulation
            COMPONENT_NAME tpc
            SOURCES testTPCGEMAmplification.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            TIMEOUT 200
            LABELS long)

o2_add_test(SAMPAProcessing
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCSimulation
            COMPONENT_NAME tpc
            SOURCES testTPCSAMPAProcessing.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(Simulation
            LABELS tpc
            PUBLIC_LINK_LIBRARIES O2::TPCSimulation
            COMPONENT_NAME tpc
            SOURCES testTPCSimulation.cxx)

# * # add the TPC run sim as a unit test (if simulation was enabled)
# * if (BUILD_SIMULATION)
# * add_test_wrap(NAME tpcsim_G4 COMMAND ${CMAKE_BINARY_DIR}/bin/o2-sim-tpc -n 2
#   -e TGeant4)
# * set_tests_properties(tpcsim_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro
#   finished succesfully")
# * add_test_wrap(NAME tpcsim_G3 COMMAND ${CMAKE_BINARY_DIR}/bin/o2-sim-tpc -n 2
#   -e TGeant3)
# * set_tests_properties(tpcsim_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro
#   finished succesfully")
# * # sets the necessary environment
# * set_tests_properties(tpcsim_G3 tpcsim_G4  PROPERTIES ENVIRONMENT
#   VMCWORKDIR=${CMAKE_SOURCE_DIR})
# * endif()
#
