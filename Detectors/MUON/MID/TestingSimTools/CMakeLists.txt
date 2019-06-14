set(MODULE_NAME "MIDTestingSimTools")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
   src/HitFinder.cxx
   src/TrackGenerator.cxx
)

set(HEADERS
   include/${MODULE_NAME}/HitFinder.h
   include/${MODULE_NAME}/TrackGenerator.h
)

Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME mid_testingSimTools_bucket)

O2_GENERATE_LIBRARY()
