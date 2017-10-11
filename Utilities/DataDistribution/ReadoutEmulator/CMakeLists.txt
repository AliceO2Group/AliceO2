# @author Gvozden Neskovic
# @brief  cmake for SubTimeBuilderDevice

o2_define_bucket(
    NAME
    ReadoutEmulatorDevice_bucket

    DEPENDENCIES
    DataDistributionCommon
    O2Device_bucket
    root_base_bucket Core Gpad Hist
)

set(MODULE_NAME "ReadoutEmulator")
set(MODULE_BUCKET_NAME ReadoutEmulatorDevice_bucket)

O2_SETUP(NAME ${MODULE_NAME})

# ReadoutEmulatorDevice lib

set(SRCS
    src/CruMemoryHandler.cxx
    src/CruEmulator.cxx
    src/ReadoutDevice.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

# ReadoutEmulatorDevice exe

set(EXE_SRC
    src/runReadoutEmulatorDevice.cxx
)

O2_GENERATE_EXECUTABLE(
    EXE_NAME "ReadoutEmulatorDevice"
    SOURCES ${EXE_SRC}
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${MODULE_BUCKET_NAME}
)
