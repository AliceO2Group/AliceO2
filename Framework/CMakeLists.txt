add_subdirectory(Foundation)

add_subdirectory(DebugGUI)

add_subdirectory(Core)

add_subdirectory(Utils)

add_subdirectory(TestWorkflows)

if(arrow_FOUND)
  add_subdirectory(ArrowTests)
endif()
