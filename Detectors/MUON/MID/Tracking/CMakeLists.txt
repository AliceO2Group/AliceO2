set(MODULE_NAME "MIDTracking")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/Tracker.cxx
   src/TrackerDevice.cxx
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_tracking_bucket)

O2_GENERATE_LIBRARY()

O2_GENERATE_EXECUTABLE(
    EXE_NAME runMIDTracking
    SOURCES src/runMIDTracking.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config/runMIDtracking.json
        DESTINATION etc/config)

add_subdirectory(test)
