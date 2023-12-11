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

# allsim is not a real library (i.e. not something that is built) but a
# convenient bag for all the deps needed in the executables below
add_library(allsim INTERFACE)

target_link_libraries(allsim
                      INTERFACE O2::SimConfig
                                O2::CommonUtils
                                O2::Steer
                                O2::SimSetup
                                O2::SimulationDataFormat
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
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::Alice3DetectorsPassive>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::ITS3Simulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::TRKSimulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::FT3Simulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::FCTSimulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::IOTOFSimulation>
                                $<$<BOOL:${ENABLE_UPGRADES}>:O2::RICHSimulation>
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
o2_add_executable(evalmat
                  COMPONENT_NAME sim
                  SOURCES o2sim_evalmat.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_add_executable(sim
                  SOURCES o2sim_parallel.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim O2::Version
                  TARGETVARNAME simdriver)


o2_add_executable(primary-server-device-runner
                  COMPONENT_NAME sim
                  SOURCES O2PrimaryServerDeviceRunner.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim
                  TARGETVARNAME simexe)
if(ENABLE_UPGRADES)
o2_add_executable(serial-run5
                  COMPONENT_NAME sim
                  SOURCES o2sim.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_add_executable(sim-run5
                  SOURCES o2sim_parallel.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim O2::Version
                  TARGETVARNAME simdriverrun5)

o2_add_executable(evalmat-run5
                  COMPONENT_NAME sim
                  SOURCES o2sim_evalmat.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)
endif()
if(NOT APPLE)
  set_property(TARGET ${simexe} PROPERTY LINK_WHAT_YOU_USE ON)
endif()

o2_add_executable(hit-merger-runner
                  COMPONENT_NAME sim
                  SOURCES O2HitMergerRunner.cxx
                  PUBLIC_LINK_LIBRARIES internal::allsim)

o2_add_executable(g4-determine-unknown-pdg-properties
                  COMPONENT_NAME sim
                  SOURCES g4DetermineUnknownPdgProperties.cxx
                  PUBLIC_LINK_LIBRARIES O2::O2TrivialMC MC::Geant4 MC::Geant4VMC O2::SimConfig)

o2_add_executable(mctracks-proxy
                  COMPONENT_NAME sim
                  SOURCES o2sim_mctracks_proxy.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::SimulationDataFormat)

o2_add_executable(kine-publisher
                  COMPONENT_NAME sim
                  SOURCES o2sim_kine_publisher.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::SimulationDataFormat O2::Steer)

o2_add_executable(dpl-eventgen
                  COMPONENT_NAME sim
                  SOURCES dpl_eventgen.cxx
                  PUBLIC_LINK_LIBRARIES O2::Generators O2::Framework O2::SimulationDataFormat)

o2_add_executable(hepmc-publisher
                  COMPONENT_NAME sim
                  SOURCES o2sim_hepmc_publisher.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::SimulationDataFormat O2::Steer MC::HepMC3)

o2_add_executable(mctracks-to-aod
                  COMPONENT_NAME sim
                  SOURCES o2sim_mctracks_to_aod.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework O2::SimulationDataFormat)

o2_add_executable(mc-to-hepmc
                  COMPONENT_NAME aod
                  SOURCES o2aod_mc_to_hepmc.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::Framework
                  O2::SimulationDataFormat
                  O2::Generators)

o2_add_dpl_workflow(mctracks-to-aod-simple-task
                    SOURCES SimExamples/McTracksToAOD/mctracks_to_aod_simple_task.cxx
                    PUBLIC_LINK_LIBRARIES O2::Framework
                    COMPONENT_NAME Analysis)

o2_data_file(COPY o2simtopology_template.json DESTINATION config)

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
if(ENABLE_UPGRADES)
o2_name_target(sim-serial-run5 NAME o2simSerialRun5Executable IS_EXE)
o2_name_target(sim-run5 NAME o2simRun5Executable IS_EXE)
o2_name_target(sim-evalmat-run5 NAME o2simEvalmatRun5Executable IS_EXE)

target_compile_definitions(${o2simSerialRun5Executable} PUBLIC SIM_RUN5)
target_compile_definitions(${o2simRun5Executable} PUBLIC SIM_RUN5)
target_compile_definitions(${o2simEvalmatRun5Executable} PUBLIC SIM_RUN5)
endif()
if (BUILD_TESTING)
o2_add_test_command(NAME o2sim_G4
                    WORKING_DIRECTORY ${SIMTESTDIR}
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
                                      --chunkSize
                                      2
                                      --skipModules
                                      MFT ZDC
                                      --configKeyValues
                                      "align-geom.mDetectors=none"
                    ENVIRONMENT "${SIMENV}"
                    LABELS "g4;sim;long")

set_tests_properties(o2sim_G4
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY" FIXTURES_SETUP
                                G4)
set_property(TEST o2sim_G4 APPEND PROPERTY ENVIRONMENT ${G4ENV})

# # note that the MT is currently only supported in the non FairMQ version
# # switched off
# o2_add_test_command(NAME o2sim_G4_mt
#                   WORKING_DIRECTORY ${SIMTESTDIR}
#                   TIMEOUT 400
#                   COMMAND $<TARGET_FILE:${o2simSerialExecutable}>
#                   COMMAND_LINE_ARGS -n
#                                      1
#                                      -e
#                                      TGeant4
#                                      --isMT
#                                      on
#                                      -o
#                                      o2simG4MT
#                   ENVIRONMENT "${SIMENV}"
#                   LABELS "g4;sim;long")
#set_tests_properties(o2sim_G4_mt
#                     PROPERTIES PASS_REGULAR_EXPRESSION
#                                "Macro finished succesfully")
#set_property(TEST o2sim_G4_mt APPEND PROPERTY ENVIRONMENT ${G4ENV})


o2_add_test(CheckStackG4
  SOURCES checkStack.cxx
  NAME o2sim_checksimkinematics_G4
  WORKING_DIRECTORY ${SIMTESTDIR}
  COMMAND_LINE_ARGS o2simG4
  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Steer O2::ITSSimulation
  NO_BOOST_TEST
  LABELS "g4;sim;long")

set_tests_properties(o2sim_checksimkinematics_G4
                     PROPERTIES FIXTURES_REQUIRED G4)

o2_add_test_command(NAME o2sim_G3
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    COMMAND $<TARGET_FILE:${o2simExecutable}>
                    COMMAND_LINE_ARGS -n
                                      2
                                      -j
                                      2
                                      -e
                                      TGeant3
                                      -o
                                      o2simG3
                                      -g
                                      pythia8pp
                                      --chunkSize
                                      10
                                      --configKeyValues
                                      "align-geom.mDetectors=none"
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
  PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat O2::Steer O2::ITSSimulation
  NO_BOOST_TEST
  LABELS "g3;sim;long")


set_tests_properties(o2sim_checksimkinematics_G3
                     PROPERTIES FIXTURES_REQUIRED G3)


o2_add_test_command(NAME o2sim_hepmc
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    TIMEOUT 400
                    COMMAND $<TARGET_FILE:${o2simExecutable}>
                    COMMAND_LINE_ARGS -n
                                       2
                                      -j
                                       2
                                      -g
                                   hepmc
                       --configKeyValues
"GeneratorFileOrCmd.fileNames=${CMAKE_SOURCE_DIR}/Generators/share/data/pythia.hepmc;HepMC.version=2;align-geom.mDetectors=none"
                                      -o
                              o2simhepmc
                    LABELS long sim hepmc3
                    ENVIRONMENT "${SIMENV}")

set_tests_properties(o2sim_hepmc
                     PROPERTIES PASS_REGULAR_EXPRESSION
                                "SIMULATION RETURNED SUCCESFULLY")
set_property(TEST o2sim_hepmc APPEND PROPERTY ENVIRONMENT ${G4ENV})

# somewhat analyse the logfiles as another means to detect problems
o2_add_test_command(NAME o2sim_G3_checklogs
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    COMMAND ${CMAKE_SOURCE_DIR}/run/simlogcheck.sh
                    COMMAND_LINE_ARGS o2simG3_serverlog o2simG3_mergerlog o2simG3_workerlog0
                    LABELS long sim hepmc3)

set_tests_properties(o2sim_G3_checklogs
                     PROPERTIES FIXTURES_REQUIRED G3)

# somewhat analyse the logfiles as another means to detect problems
o2_add_test_command(NAME o2sim_G4_checklogs
                    WORKING_DIRECTORY ${SIMTESTDIR}
                    COMMAND ${CMAKE_SOURCE_DIR}/run/simlogcheck.sh
                    COMMAND_LINE_ARGS o2simG4_serverlog o2simG4_mergerlog o2simG4_workerlog0
                    LABELS long sim
)

set_tests_properties(o2sim_G4_checklogs
                     PROPERTIES FIXTURES_REQUIRED G4)
endif()

install(FILES o2-sim-client.py PERMISSIONS GROUP_READ GROUP_EXECUTE OWNER_EXECUTE OWNER_WRITE OWNER_READ WORLD_EXECUTE WORLD_READ DESTINATION ${CMAKE_INSTALL_BINDIR})
