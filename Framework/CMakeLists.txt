add_subdirectory(Core)
add_subdirectory(TestWorkflows)
# FIXME: only run if glfw3 is detected.
if(ARROW_FOUND)
add_subdirectory(ArrowTests)
endif()
if(GLFW_FOUND)
add_subdirectory(DebugGUI)
endif()
