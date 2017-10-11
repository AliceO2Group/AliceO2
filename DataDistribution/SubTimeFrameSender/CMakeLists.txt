# @author Gvozden Neskovic
# @brief  cmake for SubTimeFrameSender

o2_define_bucket(
    NAME
    SubTimeFrameSender_bucket

    DEPENDENCIES
    DataDistributionCommon
    root_base_bucket Core Gpad Hist

    INCLUDE_DIRECTORIES
    include
)

set(MODULE_NAME "SubTimeFrameSender")
set(MODULE_BUCKET_NAME SubTimeFrameSender_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# SubTimeFrameSenderDevice exe

set(EXE_SRC
    src/SubTimeFrameSenderDevice.cxx
    src/SubTimeFrameSenderOutput.cxx
    src/runSubTimeFrameSenderDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "SubTimeFrameSenderDevice"
    SOURCES ${EXE_SRC}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)
