# @author  Barthélémy von Haller

set(MODULE_NAME "QCViewer")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/ViewerDevice.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/ViewerDevice.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_viewer_bucket)

O2_GENERATE_LIBRARY()

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runQCViewerDevice
    SOURCES src/runViewerDevice.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

if(FALSE)
  set(TEST_SRCS
      test/ViewerTestSuite.cxx
      )

  O2_GENERATE_TESTS(
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
      TEST_SRCS ${TEST_SRCS}
  )
endif()
