# @author Gvozden Neskovic
# @brief  cmake for TimeFrameBuilder

o2_define_bucket(
    NAME
    TimeFrameBuilder_bucket

    DEPENDENCIES
    DataDistributionCommon
    root_base_bucket Core Gpad Hist

    INCLUDE_DIRECTORIES
    include
)

set(MODULE_NAME "TimeFrameBuilder")
set(MODULE_BUCKET_NAME TimeFrameBuilder_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# TimeFrameBuilderDevice exe

set(EXE_SRC
    src/TimeFrameBuilderDevice.cxx
    src/TimeFrameBuilderInput.cxx
    src/runTimeFrameBuilderDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "TimeFrameBuilderDevice"
    SOURCES ${EXE_SRC}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)
