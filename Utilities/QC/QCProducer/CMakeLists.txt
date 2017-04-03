# @author  Barthélémy von Haller

set(MODULE_NAME "QCProducer")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/ProducerDevice.cxx
    src/TH1Producer.cxx
    src/TH2Producer.cxx
    src/TH3Producer.cxx
    src/THnProducer.cxx
    src/TreeProducer.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/ProducerDevice.h
    include/${MODULE_NAME}/Producer.h
    include/${MODULE_NAME}/TH1Producer.h
    include/${MODULE_NAME}/TH2Producer.h
    include/${MODULE_NAME}/TH3Producer.h
    include/${MODULE_NAME}/THnProducer.h
    include/${MODULE_NAME}/TreeProducer.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_base_bucket)

O2_GENERATE_LIBRARY()

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runQCProducerDevice
    SOURCES src/runProducer.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

# Currently broken
if(FALSE)
set(TEST_SRCS
    test/ProducerDeviceTestSuite.cxx
    test/ProducerTestSuite.cxx
    )

O2_GENERATE_TESTS(
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
    TEST_SRCS ${TEST_SRCS}
)
endif()
