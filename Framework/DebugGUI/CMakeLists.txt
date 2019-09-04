# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(GLFW_FOUND)
  set(GUI_BACKEND
      src/imgui_impl_glfw_gl3.cpp
      src/gl3w.c
      src/Sokol3DUtils.cxx
      src/GL3DUtils.cxx
      src/HandMade3DImpl.cxx
      src/DebugGUI.cxx)
  set(GUI_TARGET glfw::glfw)
else()
  set(GUI_BACKEND src/Dummy3DUtils.cxx src/DummyDebugGUI.cxx)
  set(GUI_TARGET "")
endif()

o2_add_library(DebugGUI
               SOURCES src/imgui.cpp src/imgui_draw.cpp src/imgui_extras.cpp
                       ${GUI_BACKEND}
               PUBLIC_LINK_LIBRARIES O2::FrameworkFoundation ${GUI_TARGET}
               TARGETVARNAME targetName)

target_link_libraries(${targetName} PRIVATE dl)

if(GLFW_FOUND)
  o2_add_executable(imgui
                    SOURCES
                    SOURCES test/test_ImGUI.cpp test/imgui_demo.cpp
                    PUBLIC_LINK_LIBRARIES O2::DebugGUI
                    COMPONENT_NAME Framework)
endif()

o2_add_test(ImGUIHeadless
            SOURCES test/test_ImGUIHeadless.cpp test/imgui_demo.cpp
            PUBLIC_LINK_LIBRARIES O2::DebugGUI
            COMPONENT_NAME Framework
            LABELS gui
            TIMEOUT 30)
