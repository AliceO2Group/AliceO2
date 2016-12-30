# @author Mikolaj Krzewicki

set(MODULE_NAME "O2device")

O2_SETUP(NAME ${MODULE_NAME})

#add_definitions(-DBOOST_CB_DISABLE_DEBUG -DNDEBUG)

# Define the source and header files
set(SRCS
  src/O2device.cxx
  src/SharedMemory.cxx
)

set(HEADERS
  include/${MODULE_NAME}/O2device.h
  include/${MODULE_NAME}/SharedMemory.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2device_bucket)

O2_GENERATE_LIBRARY()

