o2_add_library(TPCSimulation
               SOURCES
               src/Detector.cxx
               src/DigitMCMetaData.cxx
               src/DigitContainer.cxx
               src/DigitGlobalPad.cxx
               src/Digitizer.cxx
               src/DigitTime.cxx
               src/ElectronTransport.cxx
               src/GEMAmplification.cxx
               src/PadResponse.cxx
               src/Point.cxx
               src/SAMPAProcessing.cxx
               src/SpaceCharge.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DetectorsBase
               O2::SimulationDataFormat
               O2::TPCBase
               O2::TPCSpaceChargeBase
               ROOT::Physics)

o2_target_root_dictionary(TPCSimulation
                          HEADERS
                          include/TPCSimulation/Detector.h
                          include/TPCSimulation/DigitMCMetaData.h
                          include/TPCSimulation/DigitContainer.h
                          include/TPCSimulation/DigitGlobalPad.h
                          include/TPCSimulation/Digitizer.h
                          include/TPCSimulation/DigitTime.h
                          include/TPCSimulation/ElectronTransport.h
                          include/TPCSimulation/GEMAmplification.h
                          include/TPCSimulation/PadResponse.h
                          include/TPCSimulation/Point.h
                          include/TPCSimulation/SAMPAProcessing.h
                          include/TPCSimulation/SpaceCharge.h)

install(DIRECTORY files DESTINATION share/Detectors/TPC/)

o2_add_test(DigitContainer
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            test/testTPCDigitContainer.cxx)
o2_add_test(ElectronTransport
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            test/testTPCElectronTransport.cxx)
o2_add_test(GEMAmplification
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            test/testTPCGEMAmplification.cxx)
o2_add_test(SAMPAProcessing
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            test/testTPCSAMPAProcessing.cxx)
o2_add_test(Simulation
            LABELS
            tpc
            PUBLIC_LINK_LIBRARIES
            O2::TPCSimulation
            COMPONENT_NAME
            tpc
            SOURCES
            test/testTPCSimulation.cxx)

o2_add_test_root_macro(macro/readMCtruth.C
                       PUBLIC_LINK_LIBRARIES
                       O2::TPCBase
                       O2::SimulationDataFormat
                       LABELS
                       tpc)
#
# * # add the TPC run sim as a unit test (if simulation was enabled)
# * if (HAVESIMULATION)
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
