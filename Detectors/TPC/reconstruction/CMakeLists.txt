set(MODULE_NAME "TPCReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

link_directories( ${LINK_DIRECTORIES})

set(SRCS
   src/AdcClockMonitor.cxx 
   src/DigitData.cxx
   src/GBTFrame.cxx
   src/GBTFrameContainer.cxx
   src/HalfSAMPAData.cxx
   src/SyncPatternMonitor.cxx
)

set(HEADERS
   include/${MODULE_NAME}/AdcClockMonitor.h
   include/${MODULE_NAME}/DigitData.h
   include/${MODULE_NAME}/GBTFrame.h
   include/${MODULE_NAME}/GBTFrameContainer.h
   include/${MODULE_NAME}/HalfSAMPAData.h
   include/${MODULE_NAME}/SyncPatternMonitor.h
)
Set(LINKDEF src/TPCReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME tpc_reconstruction_bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
  tpc-read-GBTFrames
)


Set(Exe_Source
  run/readGBTFrames.cxx
)

# todo we repeat ourselves because the macro O2_GENERATE_LIBRARY dares deleting the variables we pass to it.
set(BUCKET_NAME tpc_reconstruction_bucket)
set(LIBRARY_NAME ${MODULE_NAME})

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
#  Set(EXE_NAME ${_name})
#  Set(SRCS ${_src})
#  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

