
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
  # o2sim_G4 (temporarily disabled)
  #add_test(NAME o2sim_G4 COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -m PIPE ITS TPC TRD TOF EMC PHS -e TGeant4)
  #set_tests_properties(o2sim_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  # o2sim_G4_mt (temporarily disabled)
  #add_test(NAME o2sim_G4_mt COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 10 -m PIPE ITS TPC TRD TOF EMC PHS -e TGeant4 --isMT on)
  #set_tests_properties(o2sim_G4_mt PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  #set_tests_properties(o2sim_G4_mt PROPERTIES ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR})
  # checksimkinematics_G4 (temporarily disabled)
  #add_test(NAME checksimkinematics_G4
  #  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  #  COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C)
  add_test(NAME o2sim_G3
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_BINARY_DIR}/bin/o2sim -n 2 -m PIPE ITS TPC MCH TOF EMC PHS -e TGeant3)
  set_tests_properties(o2sim_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully"
                                           ENVIRONMENT VMCWORKDIR=${CMAKE_SOURCE_DIR} FIXTURES_SETUP G3) 
  
  add_test(NAME checksimkinematics_G3
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND root -n -b -l -q ${CMAKE_SOURCE_DIR}/DataFormats/simulation/test/checkStack.C)
  set_tests_properties(checksimkinematics_G3 PROPERTIES FIXTURES_REQUIRED G3 
                                                        ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")

  #set_property(TEST checksimkinematics_G4 PROPERTY DEPENDS o2sim_G4 PASS_REGULAR_EXPRESSION "STACK TEST SUCCESSFULL")
  #set_property(TEST checksimkinematics_G4 APPEND PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")
  #set_property(TEST checksimkinematics_G4 APPEND PROPERTY ENVIRONMENT "DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/lib")
endif()
