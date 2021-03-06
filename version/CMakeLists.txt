# Generate a O2Version.h header that contains a couple of useful fields such as
# the git commit with which we build. We can also use this to define VERSION
# MACROS etc.

include(O2NameTarget)

if(DEFINED ENV{O2_GIT_REVISION})
  set(O2_GIT_REVISION $ENV{O2_GIT_REVISION})
else()
  set(O2_GIT_REVISION "--unknown--")
endif()

if(DEFINED ENV{ALIDIST_GIT_REVISION})
  set(ALIDIST_GIT_REVISION $ENV{ALIDIST_GIT_REVISION})
else()
  set(ALIDIST_GIT_REVISION "--unknown--")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/O2Version.cxx.in
               ${CMAKE_CURRENT_BINARY_DIR}/O2Version.cxx @ONLY)

add_library(version STATIC)
target_sources(version PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/O2Version.cxx)
target_include_directories(version PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(O2::Version ALIAS version)
