set(MODULE_NAME "Steer")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/O2RunAna.cxx
  src/InteractionSampler.cxx
)

set(HEADERS
  include/${MODULE_NAME}/O2RunAna.h
  include/${MODULE_NAME}/InteractionSampler.h
)

set(LINKDEF src/SteerLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_simulation_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testInteractionSampler.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
