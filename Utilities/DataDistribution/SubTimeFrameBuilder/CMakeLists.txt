# @author Gvozden Neskovic
# @brief  cmake for SubTimeBuilderDevice

o2_define_bucket(
    NAME
    SubTimeFrameBuilderDevice_bucket

    DEPENDENCIES
    DataDistributionCommon
    O2Device
    root_base_bucket Core Gpad Hist
)

set(MODULE_NAME "SubTimeFrameBuilder")
set(MODULE_BUCKET_NAME SubTimeFrameBuilderDevice_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# SubTimeFrameBuilderDevice lib

set(SRCS
    src/SubTimeFrameBuilderDevice.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

# SubTimeFrameBuilderDevice exe

set(EXE_SRC
    src/runSubTimeFrameBuilderDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "SubTimeFrameBuilderDevice"
    SOURCES ${EXE_SRC}
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)

