# @author Gvozden Neskovic
# @brief  cmake for SubTimeFrameTransporter

o2_define_bucket(
    NAME
    SubTimeFrameTransporterDevice_bucket

    DEPENDENCIES
    DataDistributionCommon
    O2Device
    root_base_bucket Core Gpad Hist
)

set(MODULE_NAME "SubTimeFrameTransporter")
set(MODULE_BUCKET_NAME SubTimeFrameTransporterDevice_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# SubTimeFrameTransporterDevice lib

set(SRCS
    src/SubTimeFrameTransporterDevice.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

# SubTimeFrameTransporterDevice exe

set(EXE_SRC
    src/runSubTimeFrameTransporterDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "SubTimeFrameTransporterDevice"
    SOURCES ${EXE_SRC}
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)
