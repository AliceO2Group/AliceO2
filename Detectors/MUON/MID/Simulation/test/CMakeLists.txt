O2_SETUP(NAME "MIDSimulationTest")
set(BUCKET_NAME mid_simulation_test_bucket)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS testGeometry.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS testSimulation.cxx
)
