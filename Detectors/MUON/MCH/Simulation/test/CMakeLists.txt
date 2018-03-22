O2_SETUP(NAME MCHSimulationTest)
set(BUCKET_NAME mch_simulation_test_bucket)

# trying to get only one test exe for this module
O2_GENERATE_EXECUTABLE(
        EXE_NAME test_MCHSimulation
        SOURCES testGeometry.cxx
        BUCKET_NAME mch_simulation_test_bucket
)

target_link_libraries(test_MCHSimulation Boost::unit_test_framework MCHSimulation)
add_test(NAME test_MCHSimulation COMMAND test_MCHSimulation)
