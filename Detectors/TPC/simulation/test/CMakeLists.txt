o2_add_test(DigitContainer
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            testTPCDigitContainer.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(ElectronTransport
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            testTPCElectronTransport.cxx)

o2_add_test(GEMAmplification
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            testTPCGEMAmplification.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            TIMEOUT 200
            LABELS slow)

o2_add_test(SAMPAProcessing
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            testTPCSAMPAProcessing.cxx
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(Simulation
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            testTPCSimulation.cxx)

o2_add_test_root_macro(macro/readMCtruth.C
                       PUBLIC_LINK_LIBRARIES
                       O2::TPCBase
                       O2::SimulationDataFormat
                       LABELS
                       tpc)
#
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
