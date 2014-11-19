GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_sim.C)

ForEach(_mcEngine IN ITEMS TGeant3 TGeant4) 
  Add_Test(run_sim_${_mcEngine} 
           ${CMAKE_BINARY_DIR}/macro/run_sim.sh 10 \"${_mcEngine}\")
  Set_Tests_Properties(run_sim_${_mcEngine} PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(run_sim_${_mcEngine} PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
EndForEach(_mcEngine IN ITEMS TGeant3 TGeant4) 

Install(FILES run_sim.C
        DESTINATION share/its
       )

