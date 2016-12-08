# @author Mikolaj Krzewicki

set(MODULE_NAME "O2device")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
  src/O2device.cxx
)

set(HEADERS
  include/${MODULE_NAME}/O2device.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME O2device_bucket)

O2_GENERATE_LIBRARY()

