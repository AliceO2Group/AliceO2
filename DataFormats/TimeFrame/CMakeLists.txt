# @author Sandro Wenzel
# @brief  cmake setup for module DataFormats/TimeFrame

set(MODULE_NAME "TimeFrame")
set(MODULE_BUCKET_NAME TimeFrame_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/TimeFrame.cxx
)

# define headers
set(HEADERS include/TimeFrame/TimeFrame.h)

set(LINKDEF src/TimeFrameLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/TimeFrameTest.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS})
