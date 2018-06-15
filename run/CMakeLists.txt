
Set(Exe_Names
  o2sim
  runTPC
  O2SimDeviceRunner
  O2PrimaryServerDeviceRunner
  O2HitMergerRunner
  o2sim_parallel
)

Set(Exe_Source
  o2sim.cxx
  runTPC.cxx
  O2SimDeviceRunner.cxx
  O2PrimaryServerDeviceRunner.cxx
  O2HitMergerRunner.cxx
  o2sim_parallel
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
  add_test(NAME o2sim_G4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -m PIPE ITS TPC TRD TOF EMC PHS -e TGeant4 -o o2simG4)
  set_tests_properties(o2sim_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully"
                                           ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR} FIXTURES_SETUP G4) 
                                           
  add_test(NAME o2sim_G4_mt 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 10 -m PIPE ITS TPC TRD TOF EMC PHS -e TGeant4 --isMT on -o o2simG4MT)
  set_tests_properties(o2sim_G4_mt PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully"
                                              ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR})
  
  add_test(NAME checksimkinematics_G4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG4.root\"\))
  set_tests_properties(checksimkinematics_G4 PROPERTIES FIXTURES_REQUIRED G4
                                                        ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")
  
  add_test(NAME o2sim_G3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -m PIPE ITS TPC MCH TOF EMC PHS -e TGeant3 -o o2simG3)
  set_tests_properties(o2sim_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully"
                                           ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR} FIXTURES_SETUP G3) 
  
  add_test(NAME checksimkinematics_G3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C\(\"o2simG3.root\"\))
  set_tests_properties(checksimkinematics_G3 PROPERTIES FIXTURES_REQUIRED G3 
                                                        ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")

# testing the parallel simulation (in preparation)
#  add_test(NAME o2sim_parallel_G4_forked
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim_parallel -n 2 -m PIPE ITS TPC TRD TOF EMC PHS -e TGeant4 -g pythia8)
#  set_tests_properties(o2sim_parallel_G4_forked PROPERTIES PASS_REGULAR_EXPRESSION "Merger process"
#                                           ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR}
#                                           FIXTURES_SETUP G4PARA
#                                           )
# we need to append more environment setting like this
#  set_property(TEST o2sim_parallel_G4_forked
#               APPEND PROPERTY ENVIRONMENT ALICE_NSIMWORKERS=1)
#  set_property(TEST o2sim_parallel_G4_forked
#               APPEND PROPERTY ENVIRONMENT ALICE_SIMINTERNALFORK=ON)

#  add_test(NAME checksimkinematics_G4PARA
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C)
#  set_tests_properties(checksimkinematics_G4PARA PROPERTIES FIXTURES_REQUIRED G4PARA
#                                                        ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")


endif()
