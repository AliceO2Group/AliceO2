  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_sim_tpc.C)

    Add_Test(run_sim_tpc_TGeant3
    ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10 \"TGeant3\")
    Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_sim_tpc_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

    Add_Test(run_sim_tpc_TGeant4
            ${CMAKE_BINARY_DIR}/macro/run_sim_tpc.sh 10 \"TGeant4\")
    Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES DEPENDS run_sim_tpc_TGeant3)
    Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_sim_tpc_TGeant4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_digi_tpc.C)

    Add_Test(run_digi_tpc_TGeant3
    ${CMAKE_BINARY_DIR}/macro/run_digi_tpc.sh 10 \"TGeant3\")
    Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES DEPENDS run_sim_TGeant3)
    Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_digi_tpc_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


    Add_Test(run_digi_tpc_TGeant4
    ${CMAKE_BINARY_DIR}/macro/run_digi_tpc.sh 10 \"TGeant4\")
    Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES DEPENDS run_sim_tpc_TGeant4)
    Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_digi_tpc_TGeant4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/run_clusterer.C)


    Add_Test(run_clusterer_TGeant3
            ${CMAKE_BINARY_DIR}/macro/run_clusterer.sh 10 \"TGeant3\")
    Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES DEPENDS run_digi_tpc_TGeant3)
    Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_clusterer_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

    Add_Test(comp_clusterer_TGeant3
            ${CMAKE_BINARY_DIR}/macro/compare_cluster.sh 10 \"TGeant3\")
    Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES DEPENDS run_clusterer_TGeant3)
    Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

    Add_Test(run_clusterer_TGeant4
    ${CMAKE_BINARY_DIR}/macro/run_clusterer.sh 10 \"TGeant4\")
    Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES DEPENDS run_digi_tpc_TGeant4)
    Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(run_clusterer_TGeant4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/test_GBTFrame.C)
  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/test_fixedPoint.C)
GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/compare_cluster.C)

  Add_Test(comp_clusterer_TGeant3
          ${CMAKE_BINARY_DIR}/macro/compare_cluster.sh 10 \"TGeant3\")
  Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES DEPENDS run_clusterer_TGeant3)
  Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES TIMEOUT "30")
  Set_Tests_Properties(comp_clusterer_TGeant3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
    Add_Test(test_GBTFrame
      ${CMAKE_BINARY_DIR}/macro/test_GBTFrame.sh)
    Set_Tests_Properties(test_GBTFrame PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(test_GBTFrame PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")

GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/load_all_libs.C)

#ITS tests with G3
  configure_file(${CMAKE_SOURCE_DIR}/macro/run_sim_its.sh ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh)
  configure_file(${CMAKE_SOURCE_DIR}/macro/run_sim_its.C ${CMAKE_BINARY_DIR}/macro/run_sim_its.C)
  configure_file(${CMAKE_SOURCE_DIR}/macro/SetIncludePath.C ${CMAKE_BINARY_DIR}/macro/SetIncludePath.C)

  add_test(NAME run_sim_its_G3 COMMAND ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh 10 TGeant3)
  set_tests_properties(run_sim_its_G3 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_sim_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


  configure_file(${CMAKE_SOURCE_DIR}/macro/run_digi_its.sh ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh)
  configure_file(${CMAKE_SOURCE_DIR}/macro/run_digi_its.C ${CMAKE_BINARY_DIR}/macro/run_digi_its.C)

  add_test(NAME run_digi_its_G3 COMMAND ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh  10 TGeant3)
  set_tests_properties(run_digi_its_G3 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_digi_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_digi_its_G3 PROPERTIES DEPENDS run_sim_its_G3)

  configure_file(${CMAKE_SOURCE_DIR}/macro/run_clus_its.sh ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh)
  configure_file(${CMAKE_SOURCE_DIR}/macro/run_clus_its.C ${CMAKE_BINARY_DIR}/macro/run_clus_its.C)

  add_test(NAME run_clus_its_G3 COMMAND ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh 10 TGeant3)
  set_tests_properties(run_clus_its_G3 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_clus_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_clus_its_G3 PROPERTIES DEPENDS run_digi_its_G3)

  configure_file(${CMAKE_SOURCE_DIR}/macro/run_trac_its.sh ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh)
  configure_file(${CMAKE_SOURCE_DIR}/macro/run_trac_its.C ${CMAKE_BINARY_DIR}/macro/run_trac_its.C)

  add_test(NAME run_trac_its_G3 COMMAND ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh 10 TGeant3)
  set_tests_properties(run_trac_its_G3 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_trac_its_G3 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_trac_its_G3 PROPERTIES DEPENDS run_clus_its_G3)



  #ITS tests with G4

  add_test(NAME run_sim_its_G4 COMMAND ${CMAKE_BINARY_DIR}/macro/run_sim_its.sh 10 TGeant4)
  set_tests_properties(run_sim_its_G4 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_sim_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")


  add_test(NAME run_digi_its_G4 COMMAND ${CMAKE_BINARY_DIR}/macro/run_digi_its.sh  10 TGeant4)
  set_tests_properties(run_digi_its_G4 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_digi_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_digi_its_G4 PROPERTIES DEPENDS run_sim_its_G4)


  add_test(NAME run_clus_its_G4 COMMAND ${CMAKE_BINARY_DIR}/macro/run_clus_its.sh 10 TGeant4)
  set_tests_properties(run_clus_its_G4 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_clus_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_clus_its_G4 PROPERTIES DEPENDS run_digi_its_G4)


  add_test(NAME run_trac_its_G4 COMMAND ${CMAKE_BINARY_DIR}/macro/run_trac_its.sh 10 TGeant4)
  set_tests_properties(run_trac_its_G4 PROPERTIES TIMEOUT "30")
  set_tests_properties(run_trac_its_G4 PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully")
  set_tests_properties(run_trac_its_G4 PROPERTIES DEPENDS run_clus_its_G4)




  GENERATE_ROOT_TEST_SCRIPT(${CMAKE_SOURCE_DIR}/macro/load_all_libs.C)
    Add_Test(load_all_libs
            ${CMAKE_BINARY_DIR}/macro/load_all_libs.sh)
    Set_Tests_Properties(load_all_libs PROPERTIES TIMEOUT "30")
    Set_Tests_Properties(load_all_libs PROPERTIES PASS_REGULAR_EXPRESSION "Macro finished succesfully.")

Install(DIRECTORY ${CMAKE_SOURCE_DIR}/macro DESTINATION share/)
