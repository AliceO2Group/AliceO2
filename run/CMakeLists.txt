# allsim is not a real library (i.e. not something that is built)
# but a convenient bag for all the deps needed in the executables below
add_library(allsim INTERFACE)

target_link_libraries(allsim
                      INTERFACE O2::SimConfig
                                O2::Steer
                                O2::SimSetup
                                FairMQ::FairMQ
                                O2::CPVSimulation
                                O2::DetectorsPassive
                                O2::EMCALSimulation
                                O2::FDDSimulation
                                O2::Field
                                O2::HMPIDSimulation
                                O2::ITSSimulation
                                O2::MCHSimulation
                                O2::MFTSimulation
                                O2::MIDSimulation
                                O2::PHOSSimulation
                                O2::T0Simulation
                                O2::TOFSimulation
                                O2::TPCSimulation
                                O2::TRDSimulation
                                O2::V0Simulation
                                O2::ZDCSimulation
                                O2::Generators)

o2_add_executable(device-runner
                  COMPONENT_NAME sim
                  SOURCES O2SimDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(serial
                  COMPONENT_NAME sim
                  SOURCES o2sim.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(sim
                  SOURCES o2sim_parallel.cxx
                  PUBLIC_LINK_LIBRARIES allsim O2::Framework)

o2_add_executable(primary-service-device-runner
                  COMPONENT_NAME sim
                  SOURCES O2PrimaryServerDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(hit-merger-runner
                  COMPONENT_NAME sim
                  SOURCES O2HitMergerRunner.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(tpc COMPONENT_NAME sim SOURCES runTPC.cxx
PUBLIC_LINK_LIBRARIES allsim O2::TPCReconstruction)

install(FILES o2simtopology.json DESTINATION share/config)

# * # add a complex simulation as a unit test (if simulation was enabled)
#   perform
# * # some checks on kinematics and track references
#
# * FIXME : use o2_add_test_wrapper for those 
#
# * add_test_wrap(NAME
# * o2sim_G4
# * WORKING_DIRECTORY
# * ${CMAKE_BINARY_DIR}
# * DONT_FAIL_ON_TIMEOUT
# * MAX_ATTEMPTS
# * 2
# * TIMEOUT
# * 400
# * COMMAND
# * ${CMAKE_BINARY_DIR}/bin/o2-sim
# * -n
# * 2
# * -j
# * 2
# * -e
# * TGeant4
# * -o
# * o2simG4)
# * set_tests_properties(o2sim_G4
# * PROPERTIES PASS_REGULAR_EXPRESSION
# * "SIMULATION RETURNED SUCCESFULLY"
# * FIXTURES_SETUP G4)
# * set_property(TEST o2sim_G4
# * APPEND
# * PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")
#
# * # note that the MT is currently only supported in the non FairMQ version
# * add_test_wrap(NAME
# * o2sim_G4_mt
# * WORKING_DIRECTORY
# * ${CMAKE_BINARY_DIR}
# * DONT_FAIL_ON_TIMEOUT
# * MAX_ATTEMPTS
# * 2
# * TIMEOUT
# * 400
# * COMMAND
# * ${CMAKE_BINARY_DIR}/bin/o2-sim-serial
# * -n
# * 1
# * -e
# * TGeant4
# * --isMT
# * on
# * -o
# * o2simG4MT)
# * set_tests_properties(o2sim_G4_mt
# * PROPERTIES PASS_REGULAR_EXPRESSION
# * "Macro finished succesfully")
#
# * add_test_wrap(
# * NAME
# * o2sim_checksimkinematics_G4
# * WORKING_DIRECTORY
# * ${CMAKE_BINARY_DIR}
# * COMMAND
# * root
# * -n
# * -b
# * -l
# * -q
# * ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG4.root
#   \"\)
# * )
# * set_tests_properties(o2sim_checksimkinematics_G4
# * PROPERTIES FIXTURES_REQUIRED G4)
# * add_test_wrap(NAME
# * o2sim_G3
# * WORKING_DIRECTORY
# * ${CMAKE_BINARY_DIR}
# * DONT_FAIL_ON_TIMEOUT
# * MAX_ATTEMPTS
# * 3
# * COMMAND
# * ${CMAKE_BINARY_DIR}/bin/o2-sim
# * -n
# * 2
# * -j
# * 2
# * -e
# * TGeant3
# * -o
# * o2simG3)
#
# * # set properties for G3 ... we use fixtures to force execution after G4
#   (since
# * # they require multiple CPUs)
# * set_tests_properties(o2sim_G3
# * PROPERTIES PASS_REGULAR_EXPRESSION
# * "SIMULATION RETURNED SUCCESFULLY"
# * FIXTURES_REQUIRED
# * G4
# * FIXTURES_SETUP
# * G3)
# * set_property(TEST o2sim_G3
# * APPEND
# * PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")
#
# * add_test_wrap(
# * NAME
# * o2sim_checksimkinematics_G3
# * WORKING_DIRECTORY
# * ${CMAKE_BINARY_DIR}
# * DONT_FAIL_ON_TIMEOUT
# * MAX_ATTEMPTS
# * 3
# * COMMAND
# * root
# * -n
# * -b
# * -l
# * -q
# * ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG3.root
#   \"\)
# * )
# * set_tests_properties(o2sim_checksimkinematics_G3
# * PROPERTIES FIXTURES_REQUIRED G3)
