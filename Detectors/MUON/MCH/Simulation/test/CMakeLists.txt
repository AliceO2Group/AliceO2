O2_SETUP(NAME MCHSimulationTest)
set(BUCKET_NAME mch_simulation_test_bucket)

O2_GENERATE_EXECUTABLE(
  EXE_NAME test_MCHSimulation
  SOURCES testDigitMerging.cxx DigitMerging.cxx testGeometry.cxx testDigitization.cxx testResponse.cxx
  BUCKET_NAME mch_simulation_test_bucket
  NO_INSTALL TRUE
)

target_link_libraries(test_MCHSimulation Boost::unit_test_framework O2MCHSimulation)
add_test_wrap(NAME test_MCHSimulation COMMAND test_MCHSimulation)

if (benchmark_FOUND)
  O2_GENERATE_EXECUTABLE(
    EXE_NAME o2-mch-bench-digit-merging
    SOURCES benchDigitMerging.cxx DigitMerging.cxx
    BUCKET_NAME ${BUCKET_NAME}
    NO_INSTALL TRUE
  )

endif()

