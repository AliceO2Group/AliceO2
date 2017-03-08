set(MODULE_NAME "LegacyRAW")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/AliDAQ.cxx
  src/AliLog.cxx
  src/AliRawData.cxx
  src/AliRawDataArray.cxx
  src/AliRawDataArrayV2.cxx
  src/AliRawDataErrorLog.cxx
  src/AliRawEquipmentHeader.cxx
  src/AliRawEventHeaderBase.cxx
  src/AliRawReaderChain.cxx
  src/AliRawReader.cxx
  src/AliRawReaderDate.cxx
  src/AliRawReaderFile.cxx
  src/AliRawReaderRoot.cxx
  src/AliRawVEquipment.cxx
  src/AliRawEquipmentV2.cxx
  src/AliRawVEvent.cxx
  src/AliRawEventV2.cxx
)

set(HEADERS
  src/AliDAQ.h
  src/AliLog.h
  src/AliRawData.h
  src/AliRawDataArray.h
  src/AliRawDataArrayV2.h
  src/AliRawDataErrorLog.h
  src/AliRawEquipmentHeader.h
  src/AliRawEventHeaderBase.h
  src/AliRawReaderChain.h
  src/AliRawReader.h
  src/AliRawReaderDate.h
  src/AliRawReaderFile.h
  src/AliRawReaderRoot.h
  src/AliRawVEquipment.h
  src/AliRawEquipmentV2.h
  src/AliRawVEvent.h
  src/AliRawEventV2.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME legacy_raw_bucket)
set(LINKDEF ${MODULE_NAME}LinkDef.h)

O2_GENERATE_LIBRARY()

#set(TEST_SRCS
#)

#O2_GENERATE_TESTS(
#  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
#  BUCKET_NAME ${BUCKET_NAME}
#  TEST_SRCS ${TEST_SRCS}
#)
