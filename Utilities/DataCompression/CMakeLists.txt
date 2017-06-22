# @author Matthias Richter
# @brief  cmake setup for module Utilities/DataCompression

set(MODULE_NAME "DataCompression")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    )

set(LIBRARY_NAME ${MODULE_NAME})

set(BUCKET_NAME common_boost_bucket)

#O2_GENERATE_LIBRARY()

Set(Exe_Names
    )

set(Exe_Source
    )

list(LENGTH Exe_Names _length)
if (LENGTH)
math(EXPR _length ${_length}-1)

ForEach (_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
EndForEach (_file RANGE 0 ${_length})
endif()

set(TEST_SRCS
  test/test_mpl_tools.cxx
  test/test_dc_primitives.cxx
  test/test_Fifo.cxx
  test/test_DataGenerator.cxx
  test/test_HuffmanCodec.cxx
  test/test_DataDeflater.cxx
)

O2_GENERATE_TESTS(
  BUCKET_NAME ${BUCKET_NAME}
  TEST_SRCS ${TEST_SRCS}
)
