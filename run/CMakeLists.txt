
Set(Exe_Names
  o2sim
)

Set(Exe_Source
  o2sim.cxx
)

set(BUCKET_NAME "run_bucket")
set(LIBRARY_NAME "")

O2_GENERATE_EXECUTABLE(
  EXE_NAME ${Exe_Names}
  SOURCES ${Exe_Source}
  MODULE_LIBRARY_NAME ${LIBRARY_NAME}
  BUCKET_NAME ${BUCKET_NAME}
)

