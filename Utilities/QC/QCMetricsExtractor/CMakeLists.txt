# @author  Barthélémy von Haller

set(MODULE_NAME "QCMetricsExtractor")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/MetricsExtractor.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/MetricsExtractor.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_base_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME QC_merger_bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runQCMetricsExtractor
    SOURCES src/runMetricsExtractor.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

# TODO add test
