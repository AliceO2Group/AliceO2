# @author Matthias Richter
# @brief  cmake setup for module devices/aliceHLTwrapper

set(MODULE_NAME "aliceHLTwrapper")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/SystemInterface.cxx
    src/HOMERFactory.cxx
    src/WrapperDevice.cxx
    src/Component.cxx
    src/MessageFormat.cxx
    src/EventSampler.cxx
    )

set(LIBRARY_NAME ${MODULE_NAME})

if (DDS_FOUND)
  set(BUCKET_NAME alicehlt_with_dds_bucket)
else ()
  set(BUCKET_NAME alicehlt_bucket)
endif ()

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
if (DDS_FOUND)
  set(BUCKET_NAME alicehlt_with_dds_bucket)
else ()
  set(BUCKET_NAME alicehlt_bucket)
endif ()

Set(Exe_Names
    aliceHLTWrapper
    aliceHLTEventSampler
    runComponent
    )

set(Exe_Source
    src/aliceHLTWrapper.cxx
    src/aliceHLTEventSampler.cxx
    src/runComponent.cxx
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
