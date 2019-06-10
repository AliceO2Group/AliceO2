if(GLFW_FOUND)
  set(GUI_BACKEND
      src/imgui_impl_glfw_gl3.cpp
      src/gl3w.c
      src/Sokol3DUtils.cxx
      src/GL3DUtils.cxx
      src/HandMade3DImpl.cxx
      src/DebugGUI.cxx)
  set(GUI_TARGET glfw)
else()
  set(GUI_BACKEND src/Dummy3DUtils.cxx src/DummyDebugGUI.cxx)
  set(GUI_TARGET "")
endif()

o2_add_library(DebugGUI
               SOURCES src/imgui.cpp src/imgui_draw.cpp src/imgui_extras.cpp
                       ${GUI_BACKEND}
               PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation ${GUI_TARGET})

if(GLFW_FOUND)
  o2_add_executable(ImGUI
                    SOURCES
                    SOURCES test/test_ImGUI.cpp test/imgui_demo.cpp
                    PUBLIC_LINK_LIBRARIES O2::DebugGUI
                    COMPONENT_NAME Framework)
endif()

o2_add_test(ImGUIHeadless
            SOURCES test/test_ImGUIHeadless.cpp test/imgui_demo.cpp
            PUBLIC_LINK_LIBRARIES O2::DebugGUI
            COMPONENT_NAME Framework
            TIMEOUT 30)
