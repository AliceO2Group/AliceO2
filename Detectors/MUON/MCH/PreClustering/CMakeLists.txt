set(MODULE_NAME "MCHPreClustering")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/DumpDigits.cxx
   src/PreClusterFinder.cxx
   src/PreClusterFinderDevice.cxx
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_preclustering_bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mch_preclustering_bucket)

# Define application
Set(Exe_Names
digit-dumper
preclusterizer
)

set(Exe_Source
src/runDumpDigits.cxx
src/runPreClusterFinder.cxx
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
BUCKET_NAME ${BUCKET_NAME}
)
EndForEach (_file RANGE 0 ${_length})

