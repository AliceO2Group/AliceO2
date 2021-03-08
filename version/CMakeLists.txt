# Generate a O2Version header and library that contain a couple of useful
# fields such as the git commit with which we build.
# We can also use this to define VERSION MACROS etc.

include(O2NameTarget)

execute_process(COMMAND git log --oneline -1 --pretty=format:%h
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} 
                OUTPUT_VARIABLE O2_GIT_REVISION
                RESULT_VARIABLE O2_GITCHECK_RETURN_STATUS)

# check if above command was ok
if (NOT O2_GITCHECK_RETURN_STATUS STREQUAL "0")
  set(O2_GIT_REVISION "--unknown--")
endif()

# check if we there are uncommited changes to the git
execute_process(COMMAND git diff-index HEAD --
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE O2_LOCAL_CHANGES
                RESULT_VARIABLE O2_LOCAL_CHANGES_RETURN_STATUS)

if (O2_LOCAL_CHANGES_RETURN_STATUS STREQUAL "0" 
    AND DEFINED O2_LOCAL_CHANGES
    AND NOT O2_LOCAL_CHANGES STREQUAL "")
  set(O2_GIT_REVISION "${O2_GIT_REVISION} (+local changes)")
endif()

if(DEFINED ENV{ALIBUILD_VERSION})
  # The build was done with alibuild
  # We record the alibuild version and alidist revision.
  # -- PLEASE KEEP comma separated format with key:value so that parsing is possible --
  set(BUILD_INFO_TEXT "ALIBUILD:$ENV{ALIBUILD_VERSION}, ALIDIST-REV:$ENV{ALIBUILD_ALIDIST_HASH}")
else()
  # The build was done by other means
  set(BUILD_INFO_TEXT "--unavailable--")
endif()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/O2Version.cxx.in
               ${CMAKE_CURRENT_BINARY_DIR}/O2Version.cxx @ONLY)

# provide a static lib for internal linking
add_library(version STATIC)
target_sources(version PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/O2Version.cxx)
target_include_directories(version PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(O2::Version ALIAS version)

# provide also a shared lib + header so that we can query from outside (ROOT macros)
add_library(O2Version_shared)
target_sources(O2Version_shared PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/O2Version.cxx)
target_include_directories(O2Version_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(O2::VersionShared ALIAS O2Version_shared)

install(FILES O2Version.h DESTINATION include)
install(TARGETS O2Version_shared DESTINATION lib)


# We also create a simple queryable json file containing the full
# alibuild/alidist information. Can be augmented with further information.

if(DEFINED ENV{ALIBUILD_VERSION})
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/version/RecordBuildDetails.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  OUTPUT_FILE o2-build-database.json)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/o2-build-database.json DESTINATION share/version/)
endif()
