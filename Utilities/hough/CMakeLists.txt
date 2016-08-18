set(MODULE_NAME "hough")

O2_SETUP(NAME ${MODULE_NAME})

Set(Exe_Names
  runHough
)

set(Exe_Source
  runHough.cxx
)

set(BUCKET_NAME hough_Bucket)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      BUCKET_NAME ${BUCKET_NAME}
  )
EndForEach(_file RANGE 0 ${_length})
