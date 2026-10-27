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
  find_package(Qt6 COMPONENTS Widgets)
  set_package_properties(Qt6 PROPERTIES TYPE OPTIONAL)
  set(GPUCA_CONFIG_GL3W 1)
endif()

# glslc compiles the shaders to SPIR-V for both backends
if(Vulkan_GLSLC_EXECUTABLE AND NOT Vulkan_GLSLC_EXECUTABLE MATCHES "NOTFOUND")
  set(GPUCA_GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
  find_program(GPUCA_GLSLC glslc)
endif()

# OpenGL backend: OpenGL library, OpenGL 4.6 headers (gl3w: from DebugGUI in O2, bundled in standalone; or GLEW) for SPIR-V shaders, and glslc
if(GPUCA_GLSLC AND TARGET OpenGL::GL)
  include(CheckSymbolExists)
  if(GPUCA_CONFIG_GL3W)
    if(ALIGPU_BUILD_TYPE STREQUAL "O2")
      get_target_property(CMAKE_REQUIRED_INCLUDES AliceO2::DebugGUI INTERFACE_INCLUDE_DIRECTORIES)
    else()
      set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
    endif()
    check_symbol_exists(GL_VERSION_4_6 "GL/glcorearb.h" GPUCA_GL3W_HAS_GL_4_6)
    if(GPUCA_GL3W_HAS_GL_4_6)
      set(GPUCA_EVENT_DISPLAY_OPENGL ON)
    endif()
  else()
    set(CMAKE_REQUIRED_INCLUDES ${GLEW_INCLUDE_DIRS})
    check_symbol_exists(GL_VERSION_4_6 "GL/glew.h" GPUCA_GLEW_HAS_GL_4_6)
    if(GPUCA_GLEW_HAS_GL_4_6)
      set(GPUCA_EVENT_DISPLAY_OPENGL ON)
    endif()
  endif()
  unset(CMAKE_REQUIRED_INCLUDES)
endif()
if(NOT GPUCA_EVENT_DISPLAY_OPENGL)
  if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    message(FATAL_ERROR "OpenGL event display needs OpenGL, OpenGL 4.6 headers and glslc")
  endif()
  message(STATUS "OpenGL event display backend disabled: needs OpenGL, OpenGL 4.6 headers and glslc")
  set(GLUT_FOUND 0)
  set(OPENGL_GLU_FOUND 0)
  set(GLEW_FOUND 0)
  set(X11_FOUND 0)
endif()

if(Vulkan_FOUND)
  string(FIND "${Vulkan_VERSION}" "." vulkan_dot_index)
  if(NOT ${Vulkan_VERSION} VERSION_GREATER_EQUAL "1.4.328" OR vulkan_dot_index EQUAL -1 OR NOT GPUCA_GLSLC)
    message(STATUS "Vulkan event display disabled: need Vulkan headers >= 1.4.328 (found ${Vulkan_VERSION}) and glslc")
    set(Vulkan_FOUND 0)
  endif()
endif()
if(Vulkan_FOUND)
  set(GPUCA_EVENT_DISPLAY_VULKAN ON)
  if(O2GPUWayland_FOUND)
    set(GPUCA_EVENT_DISPLAY_WAYLAND ON)
  endif()
endif()
if(Freetype_FOUND)
  set(GPUCA_EVENT_DISPLAY_FREETYPE ON)
  if(Fontconfig_FOUND)
    set(GPUCA_EVENT_DISPLAY_FONTCONFIG ON)
  endif()
endif()
if(Qt6_FOUND)
  set(GPUCA_EVENT_DISPLAY_QT ON)
endif()
foreach(feature OPENGL VULKAN WAYLAND FREETYPE FONTCONFIG QT)
  if(NOT GPUCA_EVENT_DISPLAY_${feature})
    set(GPUCA_EVENT_DISPLAY_${feature} OFF)
  endif()
endforeach()

set(SRCS ../utils/qsem.cxx
         GPUDisplay.cxx
         helpers/GPUDisplayMagneticField.cxx
         frontend/GPUDisplayFrontend.cxx
         frontend/GPUDisplayFrontendNone.cxx
         backend/GPUDisplayBackend.cxx
         backend/GPUDisplayBackendNone.cxx)

set(SRCS_NO_H helpers/GPUDisplayLoader.cxx
              render/GPUDisplayDraw.cxx
              render/GPUDisplayImportEvent.cxx
              helpers/GPUDisplayHelpers.cxx
              helpers/GPUDisplayAnimation.cxx
              helpers/GPUDisplayQuaternion.cxx
              helpers/GPUDisplayInterpolation.cxx
              helpers/GPUDisplayROOT.cxx
              frontend/GPUDisplayHelp.cxx
              frontend/GPUDisplayFrontendConfig.cxx)

set(HDRS_INSTALL helpers/GPUDisplayMagneticFieldConstants.h frontend/GPUDisplayFrontendKeys.h)

if(GPUCA_EVENT_DISPLAY_VULKAN)
  set(SRCS ${SRCS} backend/GPUDisplayBackendVulkan.cxx)
endif()

if(GPUCA_EVENT_DISPLAY_OPENGL)
  set(SRCS ${SRCS} backend/GPUDisplayBackendOpenGL.cxx)
  set(SRCS_NO_H ${SRCS_NO_H} helpers/GPUDisplayBackendOpenGLMagneticField.cxx)
endif()

if(GPUCA_EVENT_DISPLAY_WAYLAND)
  set(WAYLAND_PROTOCOL_HEADERS)
  set(WAYLAND_PROTOCOL_SOURCES)
  foreach(xml ${O2GPUWayland_PROTOCOLS}) # List and existence check in dependencies/FindO2GPUWayland.cmake
    get_filename_component(name ${xml} NAME_WE)
    string(REGEX REPLACE "(-unstable)?-v[0-9]+$" "" name ${name})
    set(header ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/${name}-client-protocol.h)
    set(source ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/${name}-protocol.c)
    add_custom_command(
      OUTPUT ${header}
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} client-header ${O2GPUWayland_PROTOCOLS_DIR}/${xml} ${header}
      MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/${xml}
      COMMENT "Preparing Wayland ${name}-client-protocol.h")
    add_custom_command(
      OUTPUT ${source}
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/
      COMMAND ${O2GPUWayland_SCANNER} private-code ${O2GPUWayland_PROTOCOLS_DIR}/${xml} ${source}
      MAIN_DEPENDENCY ${O2GPUWayland_PROTOCOLS_DIR}/${xml}
      COMMENT "Preparing Wayland ${name}-protocol.c")
    list(APPEND WAYLAND_PROTOCOL_HEADERS ${header})
    list(APPEND WAYLAND_PROTOCOL_SOURCES ${source})
  endforeach()

  set_property(SOURCE frontend/GPUDisplayFrontendWayland.cxx ${WAYLAND_PROTOCOL_SOURCES} APPEND PROPERTY OBJECT_DEPENDS ${WAYLAND_PROTOCOL_HEADERS})
  set_property(SOURCE frontend/GPUDisplayFrontendWayland.cxx APPEND PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/xdg-gen/)

  set(SRCS_NO_H ${SRCS_NO_H} frontend/GPUDisplayFrontendWayland.cxx ${WAYLAND_PROTOCOL_SOURCES})
endif()

if(X11_FOUND)
  set(SRCS ${SRCS} frontend/GPUDisplayFrontendX11.cxx)
endif()

if(GLUT_FOUND)
  set(SRCS ${SRCS} frontend/GPUDisplayFrontendGlut.cxx)
endif()

if(GLFW_FOUND)
  set(SRCS ${SRCS} frontend/GPUDisplayFrontendGlfw.cxx)
endif()

if(GPUCA_EVENT_DISPLAY_QT)
  qt_wrap_cpp(GPUCA_QT_MOC_SRCS frontend/GPUDisplayGUI.h) # Explicit moc / uic instead of AUTOMOC / AUTOUIC: rerun only if these files change; add new Q_OBJECT headers / .ui files here
  qt_wrap_ui(GPUCA_QT_UI_HDRS frontend/GPUDisplayGUI.ui)

  set(SRCS ${SRCS} frontend/GPUDisplayGUI.cxx frontend/GPUDisplayGUIWrapper.cxx)
  set(SRCS_NO_H ${SRCS_NO_H} ${GPUCA_QT_MOC_SRCS} ${GPUCA_QT_UI_HDRS})
  set(HDRS_INSTALL ${HDRS_INSTALL} frontend/GPUDisplayGUIChannel.h)
endif()

string(REPLACE ".cxx" ".h" HDRS "${SRCS}")

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUTracking
                 PUBLIC_INCLUDE_DIRECTORIES .
             SOURCES ${SRCS} ${SRCS_NO_H})

  o2_add_executable(field-uniform-exporter
          TARGETVARNAME exporterName
          COMPONENT_NAME gpu
          PUBLIC_LINK_LIBRARIES O2::Field O2::GPUTracking Boost::program_options
      SOURCES helpers/field-uniform-exporter.cxx helpers/GPUDisplayMagneticField.cxx)

  target_compile_definitions(${exporterName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS} ${SRCS_NO_H})
  set(targetName ${MODULE})
  add_library(O2::${MODULE} ALIAS ${MODULE})
  target_link_libraries(${targetName} PUBLIC O2::GPUTracking)
  install(TARGETS ${MODULE})
endif()

install(FILES ${HDRS} ${HDRS_INSTALL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)
install(DIRECTORY filterMacros/ DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/GPU/displayTrackFilter FILES_MATCHING PATTERN "*.C")
get_property(GPU_DISPLAY_INCLUDE_PATH DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
configure_file(filterMacros/setinclude.sh.in setinclude.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setinclude.sh PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/GPU/displayTrackFilter)

target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

message(STATUS "Building GPU Event Display (OpenGL ${GPUCA_EVENT_DISPLAY_OPENGL}, Vulkan ${GPUCA_EVENT_DISPLAY_VULKAN} ${Vulkan_VERSION}, Wayland ${GPUCA_EVENT_DISPLAY_WAYLAND}, Freetype ${GPUCA_EVENT_DISPLAY_FREETYPE}, Fontconfig ${GPUCA_EVENT_DISPLAY_FONTCONFIG}, Qt ${GPUCA_EVENT_DISPLAY_QT})")
target_include_directories(${targetName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(X11_FOUND)
  target_link_libraries(${targetName} PRIVATE X11::X11)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_X11)
endif()

if(GLUT_FOUND)
  target_link_libraries(${targetName} PRIVATE glut)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_GLUT)
endif()

if(GLFW_FOUND)
  target_link_libraries(${targetName} PRIVATE ${GLFW_LIBRARIES} glfw)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_GLFW)
endif()

if (OPENGL_GLU_FOUND)
  target_link_libraries(${targetName} PRIVATE OpenGL::GLU)
endif()

if(GPUCA_CONFIG_GL3W)
  target_compile_definitions(${targetName} PRIVATE GPUCA_DISPLAY_GL3W GPUCA_DISPLAY_OPENGL_CORE)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  target_link_libraries(${targetName} PRIVATE AliceO2::DebugGUI)
elseif(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  if(GPUCA_CONFIG_GL3W)
    target_sources(${targetName} PRIVATE ${GPUTRACKING_DIR}/display/3rdparty/gl3w.c)
  else()
    target_link_libraries(${targetName} PRIVATE ${GLEW_LIBRARIES})
  endif()
  target_sources(${targetName} PRIVATE ${GPUTRACKING_DIR}/display/3rdparty/HandMadeMath/HandMadeMathImpl.cxx)
  target_include_directories(${targetName} SYSTEM PUBLIC ${GPUTRACKING_DIR}/display/3rdparty)
endif()

if(GPUCA_EVENT_DISPLAY_OPENGL)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_OPENGL)
  target_link_libraries(${targetName} PRIVATE OpenGL::GL)
endif()

if(GPUCA_EVENT_DISPLAY_VULKAN OR GPUCA_EVENT_DISPLAY_OPENGL)
  include(shaders.cmake)
  set(SHADERS shaders/vertex.vert
              shaders/vertexPoint.vert
              shaders/vertexSmoothPoint.vert
              shaders/vertexTexture.vert
              shaders/vertexPassthrough.vert
              shaders/vertexFullscreen.vert
              shaders/fragmentDownsample.frag
              shaders/fragment.frag
              shaders/fragmentSmoothPoint.frag
              shaders/vertexSmoothLine.vert
              shaders/fragmentSmoothLine.frag
              shaders/fragmentText.frag
              shaders/fragmentTexture.frag
              shaders/field.geom)
  foreach(shader ${SHADERS})
    add_glslc_shader(${targetName} ${shader})
  endforeach()
  write_glslc_shader_header(${targetName} GPUDisplayLoadShaders.h)
endif()

if(GPUCA_EVENT_DISPLAY_VULKAN)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_VULKAN)
  target_link_libraries(${targetName} PRIVATE Vulkan::Vulkan)

  if(GPUCA_EVENT_DISPLAY_WAYLAND)
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_WAYLAND)
    target_link_libraries(${targetName} PRIVATE GPUWayland::wayland-client)
  endif()
endif()

if(GPUCA_EVENT_DISPLAY_FREETYPE)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_FREETYPE)
  target_link_libraries(${targetName} PRIVATE Freetype::Freetype)
  if(GPUCA_EVENT_DISPLAY_FONTCONFIG)
    target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_FONTCONFIG)
    target_link_libraries(${targetName} PRIVATE Fontconfig::Fontconfig)
  endif()
endif()

if(GPUCA_EVENT_DISPLAY_QT)
  target_compile_definitions(${targetName} PRIVATE GPUCA_BUILD_EVENT_DISPLAY_QT)
  target_link_libraries(${targetName} PRIVATE Qt6::Widgets)
  target_include_directories(${targetName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}) # ui_*.h
endif()

target_link_libraries(${targetName} PRIVATE TBB::tbb)
