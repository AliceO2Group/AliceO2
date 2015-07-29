GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_sim.C)

  Add_Test(run_sim_TGeant3
          ${CMAKE_BINARY_DIR}/macro/run_sim.sh 10 \"TGeant3\")
  Set_Tests_Properties(run_sim_TGeant3 PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(run_sim_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

  Add_Test(run_sim_TGeant4
          ${CMAKE_BINARY_DIR}/macro/run_sim.sh 10 \"TGeant4\")
  Set_Tests_Properties(run_sim_TGeant4 PROPERTIES DEPENDS run_sim_TGeant3)
  Set_Tests_Properties(run_sim_TGeant4 PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(run_sim_TGeant4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_digi.C)

  Add_Test(run_digi_TGeant3
          ${CMAKE_BINARY_DIR}/macro/run_digi.sh 10 \"TGeant3\")
  Set_Tests_Properties(run_digi_TGeant3 PROPERTIES DEPENDS run_sim_TGeant3)
  Set_Tests_Properties(run_digi_TGeant3 PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(run_digi_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

  Add_Test(run_digi_TGeant4
          ${CMAKE_BINARY_DIR}/macro/run_digi.sh 10 \"TGeant4\")
  Set_Tests_Properties(run_digi_TGeant4 PROPERTIES DEPENDS run_sim_TGeant4)
  Set_Tests_Properties(run_digi_TGeant4 PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(run_digi_TGeant4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/load_all_libs.C)

  Add_Test(load_all_libs
          ${CMAKE_BINARY_DIR}/macro/load_all_libs.sh)
  Set_Tests_Properties(load_all_libs PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(load_all_libs PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


Install(FILES run_sim.C run_digi.C
        DESTINATION share/its
       )

