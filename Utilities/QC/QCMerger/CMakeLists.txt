# @author  Barthélémy von Haller

set(MODULE_NAME "QCMerger")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/Merger.cxx
    src/MergerDevice.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Merger.h
    include/${MODULE_NAME}/MergerDevice.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_base_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_merger_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runQCMergerDevice
    SOURCES src/runMerger.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

set(TEST_SRCS
    test/MergerDeviceTestSuite.cxx
    test/MergerTestSuite.cxx
    )

O2_GENERATE_TESTS(
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
    TEST_SRCS ${TEST_SRCS}
)
