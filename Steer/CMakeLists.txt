set(MODULE_NAME "Steer")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/O2MCApplication.cxx
  src/InteractionSampler.cxx
  src/HitProcessingManager.cxx
)

set(HEADERS
  include/${MODULE_NAME}/InteractionSampler.h
  include/${MODULE_NAME}/HitProcessingManager.h
  include/${MODULE_NAME}/O2RunSim.h
  include/${MODULE_NAME}/O2MCApplication.h
)

set(LINKDEF src/SteerLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME steer_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testInteractionSampler.cxx
  test/testHitProcessingManager.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME steer_bucket
  TEST_SRCS ${TEST_SRCS}
)

add_subdirectory(DigitizerWorkflow)
