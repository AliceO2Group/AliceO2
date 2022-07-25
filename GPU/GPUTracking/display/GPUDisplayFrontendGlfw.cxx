// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendGlfw.cxx
/// \author David Rohr

#include "GPUDisplayFrontendGlfw.h"
#include "GPUDisplayBackend.h"
#include "GPUDisplayGUIWrapper.h"
#include "GPULogging.h"

#if defined(GPUCA_O2_LIB) && !defined(GPUCA_DISPLAY_GL3W) // Hack: we have to define this in order to initialize gl3w, cannot include the header as it clashes with glew
extern "C" int gl3wInit();
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <pthread.h>

#ifdef GPUCA_O2_LIB
#if __has_include("../src/imgui.h")
#include "../src/imgui.h"
#include "../src/imgui_impl_glfw_gl3.h"
#else
#include "DebugGUI/imgui.h"
#include "DebugGUI/imgui_impl_glfw_gl3.h"
#endif
#include <DebugGUI/DebugGUI.h>
#endif

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayFrontendGlfw::GPUDisplayFrontendGlfw()
{
  mFrontendType = TYPE_GLFW;
  mFrontendName = "GLFW";
}

static GPUDisplayFrontendGlfw* me = nullptr;

int GPUDisplayFrontendGlfw::GetKey(int key)
{
  if (key == GLFW_KEY_KP_SUBTRACT) {
    return ('-');
  }
  if (key == GLFW_KEY_KP_ADD) {
    return ('+');
  }
  if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
    return (KEY_SHIFT);
  }
  if (key == GLFW_KEY_LEFT_ALT) {
    return (KEY_ALT);
  }
  if (key == GLFW_KEY_RIGHT_ALT) {
    return (KEY_RALT);
  }
  if (key == GLFW_KEY_LEFT_CONTROL) {
    return (KEY_CTRL);
  }
  if (key == GLFW_KEY_RIGHT_CONTROL) {
    return (KEY_RCTRL);
  }
  if (key == GLFW_KEY_UP) {
    return (KEY_UP);
  }
  if (key == GLFW_KEY_DOWN) {
    return (KEY_DOWN);
  }
  if (key == GLFW_KEY_LEFT) {
    return (KEY_LEFT);
  }
  if (key == GLFW_KEY_RIGHT) {
    return (KEY_RIGHT);
  }
  if (key == GLFW_KEY_PAGE_UP) {
    return (KEY_PAGEUP);
  }
  if (key == GLFW_KEY_PAGE_DOWN) {
    return (KEY_PAGEDOWN);
  }
  if (key == GLFW_KEY_ESCAPE) {
    return (KEY_ESCAPE);
  }
  if (key == GLFW_KEY_SPACE) {
    return (KEY_SPACE);
  }
  if (key == GLFW_KEY_HOME) {
    return (KEY_HOME);
  }
  if (key == GLFW_KEY_END) {
    return (KEY_END);
  }
  if (key == GLFW_KEY_INSERT) {
    return (KEY_INSERT);
  }
  if (key == GLFW_KEY_ENTER) {
    return (KEY_ENTER);
  }
  if (key == GLFW_KEY_F1) {
    return (KEY_F1);
  }
  if (key == GLFW_KEY_F2) {
    return (KEY_F2);
  }
  if (key == GLFW_KEY_F3) {
    return (KEY_F3);
  }
  if (key == GLFW_KEY_F4) {
    return (KEY_F4);
  }
  if (key == GLFW_KEY_F5) {
    return (KEY_F5);
  }
  if (key == GLFW_KEY_F6) {
    return (KEY_F6);
  }
  if (key == GLFW_KEY_F7) {
    return (KEY_F7);
  }
  if (key == GLFW_KEY_F8) {
    return (KEY_F8);
  }
  if (key == GLFW_KEY_F9) {
    return (KEY_F9);
  }
  if (key == GLFW_KEY_F10) {
    return (KEY_F10);
  }
  if (key == GLFW_KEY_F11) {
    return (KEY_F11);
  }
  if (key == GLFW_KEY_F12) {
    return (KEY_F12);
  }
  return (0);
}

void GPUDisplayFrontendGlfw::GetKey(int key, int scancode, int mods, int& keyOut, int& keyPressOut)
{
  int specialKey = GetKey(key);
  const char* str = glfwGetKeyName(key, scancode);
  char localeKey = str ? str[0] : 0;
  if ((mods & GLFW_MOD_SHIFT) && localeKey >= 'a' && localeKey <= 'z') {
    localeKey += 'A' - 'a';
  }
  // GPUInfo("Key: key %d (%c) scancode %d -> %d (%c) special %d (%c)", key, (char)key, scancode, (int)localeKey, localeKey, specialKey, (char)specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = localeKey;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayFrontendGlfw::error_callback(int error, const char* description) { fprintf(stderr, "Error: %s\n", description); }

void GPUDisplayFrontendGlfw::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, scancode, mods, handleKey, keyPress);
  if (handleKey < 32) {
    if (action == GLFW_PRESS) {
      me->mKeys[keyPress] = true;
      me->mKeysShift[keyPress] = mods & GLFW_MOD_SHIFT;
      me->HandleKey(handleKey);
    } else if (action == GLFW_RELEASE) {
      me->mKeys[keyPress] = false;
      me->mKeysShift[keyPress] = false;
    }
  } else if (handleKey < 256) {
    if (action == GLFW_PRESS) {
      me->mLastKeyDown = handleKey;
    } else if (action == GLFW_RELEASE) {
      keyPress = me->mKeyDownMap[handleKey];
      me->mKeys[keyPress] = false;
      me->mKeysShift[keyPress] = false;
    }
  }
}

void GPUDisplayFrontendGlfw::char_callback(GLFWwindow* window, unsigned int codepoint)
{
  // GPUInfo("Key (char callback): %d %c - key: %d", codepoint, (char)codepoint, (int)me->mLastKeyDown);
  int keyPress = codepoint;
  if (keyPress >= 'a' && keyPress <= 'z') {
    keyPress += 'A' - 'a';
  }
  me->mKeyDownMap[me->mLastKeyDown] = keyPress;
  me->mKeys[keyPress] = true;
  me->mKeysShift[keyPress] = me->mKeys[KEY_SHIFT];
  me->HandleKey(codepoint);
}

void GPUDisplayFrontendGlfw::mouseButton_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (action == GLFW_PRESS) {
    if (button == 0) {
      me->mMouseDn = true;
    } else if (button == 1) {
      me->mMouseDnR = true;
    }
    me->mMouseDnX = me->mMouseMvX;
    me->mMouseDnY = me->mMouseMvY;
  } else if (action == GLFW_RELEASE) {
    if (button == 0) {
      me->mMouseDn = false;
    } else if (button == 1) {
      me->mMouseDnR = false;
    }
  }
}

void GPUDisplayFrontendGlfw::scroll_callback(GLFWwindow* window, double x, double y) { me->mMouseWheel += y * 100; }

void GPUDisplayFrontendGlfw::cursorPos_callback(GLFWwindow* window, double x, double y)
{
  me->mMouseMvX = x;
  me->mMouseMvY = y;
}

void GPUDisplayFrontendGlfw::resize_callback(GLFWwindow* window, int width, int height) { me->ResizeScene(width, height); }

#ifdef GPUCA_O2_LIB
void GPUDisplayFrontendGlfw::DisplayLoop()
{
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(me->mDisplayWidth, me->mDisplayHeight));
  ImGui::SetNextWindowBgAlpha(0.f);
  ImGui::Begin("Console", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  me->DrawGLScene();
  ImGui::End();
}
#endif

int GPUDisplayFrontendGlfw::FrontendMain()
{
  me = this;

  if (!glfwInit()) {
    fprintf(stderr, "Error initializing glfw\n");
    return (-1);
  }
  glfwSetErrorCallback(error_callback);

  glfwWindowHint(GLFW_MAXIMIZED, 1);
  if (backend()->backendType() == GPUDisplayBackend::TYPE_VULKAN) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_MIN_VERSION_MAJOR);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_MIN_VERSION_MINOR);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, mBackend->CoreProfile() ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);
#ifdef GPUCA_O2_LIB
    mUseIMGui = true;
#endif
  }
  mWindow = glfwCreateWindow(INIT_WIDTH, INIT_HEIGHT, DISPLAY_WINDOW_NAME, nullptr, nullptr);
  if (!mWindow) {
    fprintf(stderr, "Error creating glfw window\n");
    glfwTerminate();
    return (-1);
  }
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glfwMakeContextCurrent(mWindow);
  }

  glfwSetKeyCallback(mWindow, key_callback);
  glfwSetCharCallback(mWindow, char_callback);
  glfwSetMouseButtonCallback(mWindow, mouseButton_callback);
  glfwSetScrollCallback(mWindow, scroll_callback);
  glfwSetCursorPosCallback(mWindow, cursorPos_callback);
  glfwSetWindowSizeCallback(mWindow, resize_callback);
  if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
    glfwSwapInterval(1);
  }

  pthread_mutex_lock(&mSemLockExit);
  mGlfwRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

  if (mBackend->ExtInit()) {
    fprintf(stderr, "Error initializing GL extension wrapper\n");
    return (-1);
  }

#if defined(GPUCA_O2_LIB) && !defined(GPUCA_DISPLAY_GL3W)
  if (mUseIMGui && gl3wInit()) {
    fprintf(stderr, "Error initializing gl3w (2)\n");
    return (-1); // Hack: We have to initialize gl3w as well, as the DebugGUI uses it.
  }
#endif

#ifdef GPUCA_O2_LIB
  if (mUseIMGui) {
    mCanDrawText = 2;
    if (drawTextFontSize() == 0) {
      drawTextFontSize() = 12;
    }
  }
#endif

  if (InitDisplay()) {
    fprintf(stderr, "Error in GLFW display initialization\n");
    return (1);
  }

#ifdef GPUCA_O2_LIB
  if (mUseIMGui) {
    ImGui_ImplGlfwGL3_Init(mWindow, false);
    while (o2::framework::pollGUI(mWindow, DisplayLoop)) {
    }
  } else
#endif
  {
    while (!glfwWindowShouldClose(mWindow)) {
      HandleSendKey();
      if (DrawGLScene()) {
        fprintf(stderr, "Error drawing GL scene\n");
        return (1);
      }
      if (backend()->backendType() == GPUDisplayBackend::TYPE_OPENGL) {
        glfwSwapBuffers(mWindow);
      }
      glfwPollEvents();
    }
  }

  ExitDisplay();
  mDisplayControl = 2;
  pthread_mutex_lock(&mSemLockExit);
#ifdef GPUCA_O2_LIB
  if (mUseIMGui) {
    ImGui_ImplGlfwGL3_Shutdown();
  }
#endif
  glfwDestroyWindow(mWindow);
  glfwTerminate();
  mGlfwRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  return 0;
}

void GPUDisplayFrontendGlfw::DisplayExit()
{
  pthread_mutex_lock(&mSemLockExit);
  if (mGlfwRunning) {
    glfwSetWindowShouldClose(mWindow, true);
  }
  pthread_mutex_unlock(&mSemLockExit);
  while (mGlfwRunning) {
    usleep(10000);
  }
}

void GPUDisplayFrontendGlfw::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifdef GPUCA_O2_LIB
  if (mUseIMGui) {
    if (fromBotton) {
      y = ImGui::GetWindowHeight() - y;
    }
    y -= 20;
    ImGui::SetCursorPos(ImVec2(x, y));
    ImGui::TextColored(ImVec4(r, g, b, a), "%s", s);
  }
#endif
}

void GPUDisplayFrontendGlfw::SwitchFullscreen(bool set)
{
  GPUInfo("Setting Full Screen %d", (int)set);
  if (set) {
    glfwGetWindowPos(mWindow, &mWindowX, &mWindowY);
    glfwGetWindowSize(mWindow, &mWindowWidth, &mWindowHeight);
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    glfwSetWindowMonitor(mWindow, primary, 0, 0, mode->width, mode->height, mode->refreshRate);
  } else {
    glfwSetWindowMonitor(mWindow, nullptr, mWindowX, mWindowY, mWindowWidth, mWindowHeight, GLFW_DONT_CARE);
  }
}

void GPUDisplayFrontendGlfw::ToggleMaximized(bool set)
{
  if (set) {
    glfwMaximizeWindow(mWindow);
  } else {
    glfwRestoreWindow(mWindow);
  }
}

void GPUDisplayFrontendGlfw::SetVSync(bool enable) { glfwSwapInterval(enable); }

int GPUDisplayFrontendGlfw::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, FrontendThreadWrapper, this)) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
  return (0);
}

bool GPUDisplayFrontendGlfw::EnableSendKey()
{
#ifdef GPUCA_O2_LIB
  return false;
#else
  return true;
#endif
}

void GPUDisplayFrontendGlfw::getSize(int& width, int& height)
{
  glfwGetFramebufferSize(mWindow, &width, &height);
}

int GPUDisplayFrontendGlfw::getVulkanSurface(void* instance, void* surface)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  return glfwCreateWindowSurface(*(VkInstance*)instance, mWindow, nullptr, (VkSurfaceKHR*)surface) != VK_SUCCESS;
#else
  return 1;
#endif
}

unsigned int GPUDisplayFrontendGlfw::getReqVulkanExtensions(const char**& p)
{
  uint32_t glfwExtensionCount = 0;
#ifdef GPUCA_BUILD_EVENT_DISPLAY_VULKAN
  p = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
#endif
  return glfwExtensionCount;
}
