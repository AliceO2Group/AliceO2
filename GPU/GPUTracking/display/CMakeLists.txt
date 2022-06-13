# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE GPUTrackingDisplay)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  find_package(GLEW)
  set_package_properties(GLEW PROPERTIES TYPE OPTIONAL)
  find_package(X11)
  set_package_properties(X11 PROPERTIES TYPE OPTIONAL)
  find_package(GLUT)
  set_package_properties(GLUT PROPERTIES TYPE OPTIONAL)
  find_package(Vulkan)
  set_package_properties(Vulkan PROPERTIES TYPE OPTIONAL)
  find_package(Freetype)
  set_package_properties(Freetype PROPERTIES TYPE OPTIONAL)
  find_package(Fontconfig)
  set_package_properties(Fontconfig PROPERTIES TYPE OPTIONAL)
  find_package(O2GPUWayland)
  set_package_properties(O2GPUWayland PROPERTIES TYPE OPTIONAL)
  find_package(Qt5 COMPONENTS Widgets)
  set_package_properties(Qt5 PROPERTIES TYPE OPTIONAL)
endif()

if(Vulkan_FOUND AND NOT Vulkan_GLSLC_EXECUTABLE STREQUAL "Vulkan_GLSLC_EXECUTABLE-NOTFOUND")
  include(../Standalone/cmake/vulkan_display.cmake)
  set(GPUCA_EVENT_DISPLAY_VULKAN ON)
  if(O2GPUWayland_FOUND)
    set(GPUCA_EVENT_DISPLAY_WAYLAND ON)
  endif()
endif()
if(Freetype_FOUND)
  set(GPUCA_EVENT_DISPLAY_FREETYPE ON)
endif()
if(Qt5_FOUND)
  set(GPUCA_EVENT_DISPLAY_QT ON)
endif()

set(SRCS ../utils/qsem.cxx
         GPUDisplay.cxx
         GPUDisplayFrontend.cxx
         GPUDisplayBackend.cxx
         GPUDisplayBackendOpenGL.cxx
         GPUDisplayFrontendGlfw.cxx)

set(SRCS_NO_H GPUDisplayLoader.cxx
              GPUDisplayQuaternion.cxx
              GPUDisplayInterpolation.cxx
              GPUDisplayKeys.cxx
              GPUDisplayROOT.cxx)

set(HDRS_INSTALL GPUDisplayShaders.h)

if(GPUCA_EVENT_DISPLAY_VULKAN)
  set(SRCS ${SRCS} GPUDisplayBackendVulkan.cxx)
  if(GPUCA_EVENT_DISPLAY_WAYLAND)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-client-protocol.h
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} client-header ${O2GPUWayland_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-client-protocol.h
  MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-protocol.c
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} private-code ${O2GPUWayland_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-protocol.c
  MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-client-protocol.h
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} client-header ${O2GPUWayland_PROTOCOLS_DIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-client-protocol.h
  MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-protocol.c
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} private-code ${O2GPUWayland_PROTOCOLS_DIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-protocol.c
  MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml)
  
    set_property(SOURCE GPUDisplayFrontendWayland.cxx ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-protocol.c APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-client-protocol.h ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-client-protocol.h)
    set_property(SOURCE GPUDisplayFrontendWayland.cxx APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/)
    
    set(SRCS_NO_H ${SRCS_NO_H} GPUDisplayFrontendWayland.cxx ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-shell-protocol.c ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/xdg-decoration-protocol.c)
  endif()
endif()

if(X11_FOUND)
  set(SRCS ${SRCS} GPUDisplayFrontendX11.cxx)
endif()

if(GLUT_FOUND)
  set(SRCS ${SRCS} GPUDisplayFrontendGlut.cxx)
endif()

if(GPUCA_EVENT_DISPLAY_QT)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_AUTOUIC ON)
  
  set(SRCS ${SRCS} GPUDisplayGUI.cxx GPUDisplayGUIWrapper.cxx)
  set(SRCS_NO_H ${SRCS_NO_H} GPUDisplayGUI.ui)
endif()

string(REPLACE ".cxx" ".h" HDRS "${SRCS}")

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUTracking AliceO2::DebugGUI
                 PUBLIC_INCLUDE_DIRECTORIES .
             SOURCES ${SRCS} ${SRCS_NO_H})

  target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_GLFW GPUCA_DISPLAY_GL3W GPUCA_DISPLAY_OPENGL_CORE)
  
  install(FILES ${HDRS} ${HDRS_INSTALL} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS} ${SRCS_NO_H})
  set(targetName ${MODULE})
  add_library(O2::${MODULE} ALIAS ${MODULE})
  target_link_libraries(${targetName} PUBLIC O2::GPUTracking)
  install(TARGETS ${MODULE})

  install(DIRECTORY filterMacros/ DESTINATION displayTrackFilter FILES_MATCHING PATTERN "*.C")
  get_property(GPU_DISPLAY_INCLUDE_PATH DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  configure_file(filterMacros/setinclude.sh.in setinclude.sh @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setinclude.sh PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE DESTINATION displayTrackFilter)
endif()

message(STATUS "Building GPU Event Display (Vulkan ${GPUCA_EVENT_DISPLAY_VULKAN}, Wayland ${GPUCA_EVENT_DISPLAY_WAYLAND}, Freetype ${GPUCA_EVENT_DISPLAY_FREETYPE}, Fontconfig ${Fontconfig_FOUND}, Qt ${GPUCA_EVENT_DISPLAY_QT})")
target_link_libraries(${targetName} PUBLIC ${GLFW_LIBRARIES} OpenGL::GL)

if(X11_FOUND)
  target_link_libraries(${targetName} PRIVATE X11::X11)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_X11)
endif()

if(GLUT_FOUND)
  target_link_libraries(${targetName} PRIVATE glut)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_GLUT)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  if(GLFW_FOUND)
    target_link_libraries(${targetName} PRIVATE glfw)
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_GLFW)
  endif()
  if(CONFIG_GL3W)
    target_compile_definitions(${targetName} PRIVATE GPUCA_DISPLAY_GL3W)
  else()
    target_link_libraries(${targetName} PRIVATE ${GLEW_LIBRARIES})
  endif()
endif()

if (OPENGL_GLU_FOUND)
  target_link_libraries(${targetName} PRIVATE OpenGL::GLU)
endif()

if(GPUCA_EVENT_DISPLAY_VULKAN)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_VULKAN)
  target_link_libraries(${targetName} PRIVATE Vulkan::Vulkan)

  add_glslc_shader(${targetName} shaders/fragment.frag)
  add_glslc_shader(${targetName} shaders/vertex.vert)
  add_glslc_shader(${targetName} shaders/fragmentText.frag)
  add_glslc_shader(${targetName} shaders/vertexTexture.vert)
  add_glslc_shader(${targetName} shaders/fragmentTexture.frag)
  add_glslc_shader(${targetName} shaders/fragmentUniform.frag)
  add_glslc_shader(${targetName} shaders/vertexPoint.vert)
  if(GPUCA_EVENT_DISPLAY_WAYLAND)
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_WAYLAND)
    target_link_libraries(${targetName} PRIVATE GPUWayland::wayland-client)
  endif()
endif()

if(GPUCA_EVENT_DISPLAY_FREETYPE)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_FREETYPE)
  target_link_libraries(${targetName} PRIVATE Freetype::Freetype)
  if(Fontconfig_FOUND)
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG)
    target_link_libraries(${targetName} PRIVATE Fontconfig::Fontconfig)
  endif()
endif()

if(GPUCA_EVENT_DISPLAY_QT)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_QT)
  target_link_libraries(${targetName} PRIVATE Qt5::Widgets)
endif()

if(OpenMP_CXX_FOUND)
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
