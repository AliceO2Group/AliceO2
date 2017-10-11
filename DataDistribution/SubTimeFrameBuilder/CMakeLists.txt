# @author Gvozden Neskovic
# @brief  cmake for SubTimeFrameBuilder

o2_define_bucket(
    NAME
    SubTimeFrameBuilder_bucket

    DEPENDENCIES
    DataDistributionCommon
    root_base_bucket Core Gpad Hist

    INCLUDE_DIRECTORIES
    include
)

set(MODULE_NAME "StfBuilder")
set(MODULE_BUCKET_NAME SubTimeFrameBuilder_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# SubTimeFrameBuilder exe

set(EXE_SRC
    src/SubTimeFrameBuilderInput.cxx
    src/SubTimeFrameBuilderDevice.cxx
    src/runSubTimeFrameBuilderDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "SubTimeFrameBuilderDevice"
    SOURCES ${EXE_SRC}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)

# man pages
O2_GENERATE_MAN(NAME SubTimeFrameBuilderDevice)
