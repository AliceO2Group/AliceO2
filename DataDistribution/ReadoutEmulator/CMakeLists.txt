# @author Gvozden Neskovic
# @brief  cmake for SubTimeBuilderDevice

o2_define_bucket(
    NAME
    ReadoutEmulator_bucket

    DEPENDENCIES
    DataDistributionCommon
    root_base_bucket Core Gpad Hist

    INCLUDE_DIRECTORIES
    include
)

set(MODULE_NAME "ReadoutEmulator")
set(MODULE_BUCKET_NAME ReadoutEmulator_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# ReadoutEmulatorDevice exe

set(EXE_SRC
    src/CruMemoryHandler.cxx
    src/CruEmulator.cxx
    src/ReadoutDevice.cxx
    src/runReadoutEmulatorDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "ReadoutEmulatorDevice"
    SOURCES ${EXE_SRC}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)
