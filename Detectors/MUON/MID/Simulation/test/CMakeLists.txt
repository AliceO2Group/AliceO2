O2_SETUP(NAME "MIDSimulationTest")
set(BUCKET_NAME mid_simulation_test_bucket)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS testGeometry.cxx
)