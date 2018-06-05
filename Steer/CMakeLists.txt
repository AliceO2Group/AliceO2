set(MODULE_NAME "Steer")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/InteractionSampler.cxx
  src/HitProcessingManager.cxx
)

set(HEADERS
  include/${MODULE_NAME}/InteractionSampler.h
  include/${MODULE_NAME}/HitProcessingManager.h
)

set(LINKDEF src/SteerLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_simulation_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testInteractionSampler.cxx
  test/testHitProcessingManager.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME data_format_simulation_test_bucket
  TEST_SRCS ${TEST_SRCS}
)

add_subdirectory(DigitizerWorkflow)
