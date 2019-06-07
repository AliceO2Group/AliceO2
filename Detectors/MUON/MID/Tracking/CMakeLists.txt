o2_add_library(MIDTracking SOURCES src/Tracker.cxx
        PUBLIC_LINK_LIBRARIES O2::DataFormatsMID
        O2::MIDBase
        O2::MIDTestingSimTools)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
