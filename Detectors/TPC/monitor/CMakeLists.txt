set(MODULE_NAME "TPCMonitor")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/SimpleEventDisplay.cxx
)

set(HEADERS
  include/${MODULE_NAME}/SimpleEventDisplay.h
)

Set(LINKDEF src/${MODULE_NAME}LinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_monitor_bucket)

O2_GENERATE_LIBRARY()

#install(
  #DIRECTORY files
  #DESTINATION share/Detectors/TPC/
#)
# ===| Tests |==================================================================
#set(TEST_SRCS
   #test/testTPCBase.cxx
#)

#O2_GENERATE_TESTS(
  #BUCKET_NAME ${BUCKET_NAME}
  #MODULE_LIBRARY_NAME ${MODULE_NAME}
  #TEST_SRCS ${TEST_SRCS}
#)

# ===| generate executables |===================================================
Set(Exe_Names
  tpc-monitor
)

Set(Exe_Source
  run/runMonitor.cxx
)

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(BUCKET_NAME tpc_monitor_bucket)
set(LIBRARY_NAME ${MODULE_NAME})

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  # FIXME: propably a copy paste remnant, remove?
#  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

