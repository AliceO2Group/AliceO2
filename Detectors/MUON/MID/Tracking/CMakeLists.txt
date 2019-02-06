set(MODULE_NAME "MIDTracking")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Tracker.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_tracking_bucket)

O2_GENERATE_LIBRARY()

add_subdirectory(test)
