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
				                        O2::MCHBase
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
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::Alice3DetectorsPassive>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::ITS3Simulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::TRKSimulation>
                                O2::Generators)

add_library(internal::allsim ALIAS allsim)

o2_add_executable(device-runner
                  COMPONENT_NAME sim
                  SOURCES O2SimDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_add_executable(serial
                  COMPONENT_NAME sim
                  SOURCES o2sim.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_add_executable(sim
                  SOURCES o2sim_parallel.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim O2::Framework)

o2_add_executable(primary-server-device-runner
                  COMPONENT_NAME sim
                  SOURCES O2PrimaryServerDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim
                  TARGETVARNAME simexe)
if(NOT APPLE)
  set_property(TARGET ${simexe} PROPERTY LINK_WHAT_YOU_USE ON)
endif()

o2_add_executable(hit-merger-runner
                  COMPONENT_NAME sim
                  SOURCES O2HitMergerRunner.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_data_file(COPY o2simtopology.json DESTINATION config)

# * # add a complex simulation as a unit test (if simulation was enabled)
#   perform
# * # some checks on kinematics and track references


# make workspace for simulation tests
set(SIMTESTDIR ${CMAKE_BINARY_DIR}/o2sim_tests)
file(MAKE_DIRECTORY ${SIMTESTDIR})

# define environment needed for simulation tests (during build phase)
# needed because the build environment is different to runtime environment
set(SIMENV "ALICE_O2SIM_DUMPLOG=ON;O2_ROOT=${CMAKE_BINARY_DIR}/stage;VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/share;ROOT_INCLUDE_PATH=$ENV{ROOT_INCLUDE_PATH}:$ENV{HEPMC3_ROOT}/include;ROOT_DYN_PATH=$ENV{ROOTSYS}/lib")
message(STATUS "SIMENV = ${SIMENV}")

o2_name_target(sim NAME o2simExecutable IS_EXE)
o2_name_target(sim-serial NAME o2simSerialExecutable IS_EXE)

o2_add_test_wrapper(NAME o2sim_G4
                    WORKING_DIRECTORY ${SIMTESTDIR}
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
                    ENVIRONMENT "${SIMENV}"
                    LABELS "g4;sim;long")

set_tests_properties(o2sim_G4
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY" FIXTURES_SETUP
                                G4)
set_property(TEST o2sim_G4 APPEND PROPERTY ENVIRONMENT ${G4ENV})

# # note that the MT is currently only supported in the non FairMQ version
o2_add_test_wrapper(NAME o2sim_G4_mt
                    WORKING_DIRECTORY ${SIMTESTDIR}
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
                   ENVIRONMENT "${SIMENV}"
                   LABELS "g4;sim;long")
set_tests_properties(o2sim_G4_mt
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "Macro finished succesfully")
set_property(TEST o2sim_G4_mt APPEND PROPERTY ENVIRONMENT ${G4ENV})


o2_add_test(CheckStackG4
  SOURCES checkStack.cxx
  NAME o2sim_checksimkinematics_G4
  WORKING_DIRECTORY ${SIMTESTDIR}
  COMMAND_LINE_ARGS o2simG4
  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Steer
  NO_BOOST_TEST
  LABELS "g4;sim;long")

set_tests_properties(o2sim_checksimkinematics_G4
                     PROPERTIES FIXTURES_REQUIRED G4)

o2_add_test_wrapper(NAME o2sim_G3
                    WORKING_DIRECTORY ${SIMTESTDIR}
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
                    ENVIRONMENT "${SIMENV}"
)

# set properties for G3 ... we use fixtures to force execution after G4 (since
# they require multiple CPUs)
set_tests_properties(o2sim_G3
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY"
                                FIXTURES_REQUIRED
                                G4
                                FIXTURES_SETUP
                                G3)

o2_add_test(CheckStackG3
  SOURCES checkStack.cxx
  NAME o2sim_checksimkinematics_G3
  WORKING_DIRECTORY ${SIMTESTDIR}
  COMMAND_LINE_ARGS o2simG3
  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Steer
  NO_BOOST_TEST
  LABELS "g3;sim;long")


set_tests_properties(o2sim_checksimkinematics_G3
                     PROPERTIES FIXTURES_REQUIRED G3)


o2_add_test_wrapper(NAME o2sim_hepmc
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    DONT_FAIL_ON_TIMEOUT
                    MAX_ATTEMPTS 2
                    TIMEOUT 400
                    COMMAND $<TARGET_FILE:${o2simExecutable}>
                    COMMAND_LINE_ARGS -n
                                      2
                                     -j
                                      2
				      -g
				      hepmc
				      --configKeyValues
				      "HepMC.fileName=${CMAKE_SOURCE_DIR}/Generators/share/data/pythia.hepmc;HepMC.version=2"
                                      -o
                                      o2simhepmc
                    LABELS long sim hepmc3
                    ENVIRONMENT "${SIMENV}")

set_tests_properties(o2sim_hepmc
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY")

# somewhat analyse the logfiles as another means to detect problems
o2_add_test_wrapper(NAME o2sim_G3_checklogs
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    DONT_FAIL_ON_TIMEOUT
                    COMMAND ${CMAKE_SOURCE_DIR}/run/simlogcheck.sh
                    COMMAND_LINE_ARGS o2simG3_serverlog o2simG3_mergerlog o2simG3_workerlog0
                    LABELS long sim hepmc3)

set_tests_properties(o2sim_G3_checklogs
                     PROPERTIES FIXTURES_REQUIRED G3)

# somewhat analyse the logfiles as another means to detect problems
o2_add_test_wrapper(NAME o2sim_G4_checklogs
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    DONT_FAIL_ON_TIMEOUT
                    COMMAND ${CMAKE_SOURCE_DIR}/run/simlogcheck.sh
                    COMMAND_LINE_ARGS o2simG4_serverlog o2simG4_mergerlog o2simG4_workerlog0
                                       LABELS long sim
)

set_tests_properties(o2sim_G3_checklogs
                     PROPERTIES FIXTURES_REQUIRED G4)
