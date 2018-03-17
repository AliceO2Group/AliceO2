O2_SETUP(NAME MIDTrackingTest)
set(BUCKET_NAME mid_tracking_test_bucket)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS testTracker.cxx
)

if (benchmark_FOUND)
    O2_GENERATE_EXECUTABLE(
      EXE_NAME benchTracker
      SOURCES bench_Tracker.cxx
      BUCKET_NAME ${BUCKET_NAME})
endif ()
