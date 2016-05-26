O2_SETUP(NAME "flp2epn")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/startFLP2EPN.sh.in ${CMAKE_BINARY_DIR}/bin/startFLP2EPN.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/startMerger.sh.in ${CMAKE_BINARY_DIR}/bin/startMerger.sh)


set(HEADERS
    include/O2FLPex.h
    include/O2EPNex.h
    include/O2Proxy.h
    include/O2Merger.h
    include/O2EpnMerger.h
    )

set(SRCS
    src/O2FLPex.cxx
    src/O2EPNex.cxx
    src/O2Proxy.cxx
    src/O2Merger.cxx
    src/O2EpnMerger.cxx
    )

set(LIBRARY_NAME FLP2EPNex)
set(BUCKET_NAME flp2epn_Bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
    ${Exe_Names}
    testFLP
    testEPN
    testEPN_M
    testProxy
    testMerger
    )

set(Exe_Source
    src/runFLP.cxx
    src/runEPN.cxx
    src/runEPN_M.cxx
    src/runProxy.cxx
    src/runMerger.cxx
    )

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(LIBRARY_NAME FLP2EPNex)
set(BUCKET_NAME flp2epn_Bucket)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

ForEach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
EndForEach (_file RANGE 0 ${_length})
