Set(Exe_Names
  o2sim_serial
  runTPC
  O2SimDeviceRunner
  O2PrimaryServerDeviceRunner
  O2HitMergerRunner
  o2sim
)

Set(Exe_Source
  o2sim.cxx
  runTPC.cxx
  O2SimDeviceRunner.cxx
  O2PrimaryServerDeviceRunner.cxx
  O2HitMergerRunner.cxx
  o2sim_parallel.cxx
)

set(BUCKET_NAME "run_bucket")
set(LIBRARY_NAME "")

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

Install(FILES o2simtopology.json DESTINATION share/config)

# add a complex simulation as a unit test (if simulation was enabled)
# perform some checks on kinematics and track references
if (HAVESIMULATION)

  add_test_wrap(NAME o2sim_G4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DONT_FAIL_ON_TIMEOUT
    MAX_ATTEMPTS 2
    TIMEOUT 400
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -j 2 -e TGeant4 -o o2simG4)
  set_tests_properties(o2sim_G4 PROPERTIES PASS_REGULAR_EXPRESSION "SIMULATION RETURNED SUCCESFULLY"
                                           FIXTURES_SETUP G4)
  set_property(TEST o2sim_G4 APPEND PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")

  # note that the MT is currently only supported in the non FairMQ version
  add_test_wrap(NAME o2sim_G4_mt
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DONT_FAIL_ON_TIMEOUT
    MAX_ATTEMPTS 2
    TIMEOUT 400
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim_serial -n 1 -e TGeant4 --isMT on -o o2simG4MT)
  set_tests_properties(o2sim_G4_mt PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

  add_test_wrap(NAME o2sim_checksimkinematics_G4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG4.root\"\))
  set_tests_properties(o2sim_checksimkinematics_G4 PROPERTIES FIXTURES_REQUIRED G4)
  add_test_wrap(NAME o2sim_G3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DONT_FAIL_ON_TIMEOUT
    MAX_ATTEMPTS 3
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -j 2 -e TGeant3 -o o2simG3)

  # set properties for G3 ... we use fixtures to force execution after G4 (since they require multiple CPUs)
  set_tests_properties(o2sim_G3 PROPERTIES PASS_REGULAR_EXPRESSION "SIMULATION RETURNED SUCCESFULLY"
                                           FIXTURES_REQUIRED G4
                                           FIXTURES_SETUP G3)
  set_property(TEST o2sim_G3 APPEND PROPERTY ENVIRONMENT "ALICE_O2SIM_DUMPLOG=ON")

  add_test_wrap(NAME o2sim_checksimkinematics_G3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DONT_FAIL_ON_TIMEOUT
    MAX_ATTEMPTS 3
    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG3.root\"\))
  set_tests_properties(o2sim_checksimkinematics_G3 PROPERTIES FIXTURES_REQUIRED G3)
endif()
