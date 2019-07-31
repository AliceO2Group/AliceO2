# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# allsim is not a real library (i.e. not something that is built) but a
# convenient bag for all the deps needed in the executables below
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
                                O2::FT0Simulation
                                O2::TOFSimulation
                                O2::TPCSimulation
                                O2::TRDSimulation
                                O2::FV0Simulation
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

o2_add_executable(primary-server-device-runner
                  COMPONENT_NAME sim
                  SOURCES O2PrimaryServerDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(hit-merger-runner
                  COMPONENT_NAME sim
                  SOURCES O2HitMergerRunner.cxx
                  PUBLIC_LINK_LIBRARIES allsim)

o2_add_executable(tpc
                  COMPONENT_NAME sim
                  SOURCES runTPC.cxx
                  PUBLIC_LINK_LIBRARIES allsim O2::TPCReconstruction)

o2_data_file(COPY o2simtopology.json DESTINATION config)

# * # add a complex simulation as a unit test (if simulation was enabled)
#   perform
# * # some checks on kinematics and track references

o2_name_target(sim NAME o2simExecutable IS_EXE)
o2_name_target(sim-serial NAME o2simSerialExecutable IS_EXE)

o2_add_test_wrapper(NAME o2sim_G4
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    DONT_FAIL_ON_TIMEOUT
                    MAX_ATTEMPTS 2
                    TIMEOUT 400
                    COMMAND $<TARGET_FILE:${o2simExecutable}>
                    COMMAND_LINE_ARGS -n
                                      2
                                      -j
                                      2
                                      -e
                                      TGeant4
                                      -o
                                      o2simG4
                    LABELS long sim g4
                    ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

set_tests_properties(o2sim_G4
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY" FIXTURES_SETUP
                                G4)
set_property(TEST o2sim_G4 APPEND PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")

# # note that the MT is currently only supported in the non FairMQ version
o2_add_test_wrapper(NAME o2sim_G4_mt
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    DONT_FAIL_ON_TIMEOUT
                    MAX_ATTEMPTS 2
                    TIMEOUT 400
                    COMMAND $<TARGET_FILE:${o2simSerialExecutable}>
                    COMMAND_LINE_ARGS -n
                                      1
                                      -e
                                      TGeant4
                                      --isMT
                                      on
                                      -o
                                      o2simG4MT
                    ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)
set_tests_properties(o2sim_G4_mt
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "Macro finished succesfully")

o2_add_test_wrapper(
  NAME o2sim_checksimkinematics_G4
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND ${ROOT_root_CMD}
  COMMAND_LINE_ARGS
    -n -b -l -q
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG4.root\"\)
  LABELS g4 sim)

set_tests_properties(o2sim_checksimkinematics_G4
                     PROPERTIES FIXTURES_REQUIRED G4)

o2_add_test_wrapper(NAME o2sim_G3
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    DONT_FAIL_ON_TIMEOUT
                    MAX_ATTEMPTS 3
                    COMMAND $<TARGET_FILE:${o2simExecutable}>
                    COMMAND_LINE_ARGS -n
                                      2
                                      -j
                                      2
                                      -e
                                      TGeant3
                                      -o
                                      o2simG3
                    LABELS g3 sim long
                    ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

# set properties for G3 ... we use fixtures to force execution after G4 (since
# they require multiple CPUs)
set_tests_properties(o2sim_G3
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY"
                                FIXTURES_REQUIRED
                                G4
                                FIXTURES_SETUP
                                G3)
set_property(TEST o2sim_G3 APPEND PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")

o2_add_test_wrapper(
  NAME o2sim_checksimkinematics_G3
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DONT_FAIL_ON_TIMEOUT
  MAX_ATTEMPTS 3
  COMMAND ${ROOT_root_CMD}
  COMMAND_LINE_ARGS
    -n -b -l -q
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG3.root\"\)
  LABELS g3 sim)

set_tests_properties(o2sim_checksimkinematics_G3
                     PROPERTIES FIXTURES_REQUIRED G3)
