# @author Matthias Richter
# @brief  cmake setup for module Utilities/DataFlow

set(MODULE_NAME "DataFlow")
# the bucket contains the following dependencies
#  - common_boost_bucket
#  - Base
#  - Headers
#  - O2device
#  - dl
# the 'dl' dependency is needed as the device boilerplate code in
# runSimpleMQStateMachine.h uses dlopen etc. Probably this hidden
# dependency can be avoided by including the to some compiled FairMQ
# library

set(MODULE_BUCKET_NAME O2DeviceApplication_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/FakeTimeframeBuilder.cxx
    src/FakeTimeframeGeneratorDevice.cxx
    src/HeartbeatSampler.cxx
    src/SubframeBuilderDevice.cxx
    src/TimeframeParser.cxx
    src/TimeframeReaderDevice.cxx
    src/TimeframeValidatorDevice.cxx
    src/TimeframeWriterDevice.cxx
    src/EPNReceiverDevice.cxx
    src/FLPSenderDevice.cxx
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

O2_GENERATE_LIBRARY()

# TODO: feature of macro, it deletes the variables we pass to it, set them again
# this has to be fixed in the macro implementation
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ${MODULE_BUCKET_NAME})

Set(Exe_Names
    FakeTimeframeGeneratorDevice
    heartbeatSampler
    SubframeBuilderDevice
    TimeframeReaderDevice
    TimeframeValidatorDevice
    TimeframeWriterDevice
    EPNReceiverDevice
    FLPSenderDevice
   )

set(Exe_Source
    src/runFakeTimeframeGeneratorDevice.cxx
    src/runHeartbeatSampler.cxx
    src/runSubframeBuilderDevice.cxx
    src/runTimeframeReaderDevice.cxx
    src/runTimeframeValidatorDevice.cxx
    src/runTimeframeWriterDevice.cxx
    src/runEPNReceiver.cxx
    src/runFLPSender.cxx
    )

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach (_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${MODULE_BUCKET_NAME}
  )
EndForEach (_file RANGE 0 ${_length})


O2_GENERATE_EXECUTABLE(
  EXE_NAME TimeframeValidationTool
  SOURCES src/TimeframeValidationTool
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${MODULE_BUCKET_NAME}
)

set(TEST_SRCS
  test/test_TimeframeParser.cxx
  test/test_SubframeUtils01.cxx
  test/test_PayloadMerger01.cxx
)

O2_GENERATE_TESTS(
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS})

O2_GENERATE_MAN(NAME TimeframeReaderDevice)
O2_GENERATE_MAN(NAME TimeframeWriterDevice)
O2_GENERATE_MAN(NAME SubframeBuilderDevice)
