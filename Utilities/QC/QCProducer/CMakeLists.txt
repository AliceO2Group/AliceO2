# @author  Barthélémy von Haller

set(MODULE_NAME "QCProducer")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/HistogramProducer.cxx
    src/ProducerDevice.cxx
    src/TreeProducer.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/HistogramProducer.h
    include/${MODULE_NAME}/ProducerDevice.h
    include/${MODULE_NAME}/TreeProducer.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_base_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_base_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runProducer
    SOURCES src/runProducer.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

# TODO add test
