// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendGlfw.cxx
/// \author David Rohr

// GL EXT must be the first header
#include "GPUDisplayExt.h"

#include "GPUDisplayBackendGlfw.h"
#include "GPULogging.h"

#if defined(GPUCA_O2_LIB) && !defined(GPUCA_DISPLAY_GL3W) // Hack: we have to define this in order to initialize gl3w, cannot include the header as it clashes with glew
extern "C" int gl3wInit();
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

static GPUDisplayBackendGlfw* me = nullptr;

int GPUDisplayBackendGlfw::GetKey(int key)
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
  if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
    return (KEY_ALT);
  }
  if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
    return (KEY_CTRL);
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

void GPUDisplayBackendGlfw::GetKey(int key, int scancode, int mods, int& keyOut, int& keyPressOut)
{
  int specialKey = GetKey(key);
  const char* str = glfwGetKeyName(key, scancode);
  char localeKey = str ? str[0] : 0;
  if ((mods & GLFW_MOD_SHIFT) && localeKey >= 'a' && localeKey <= 'z') {
    localeKey += 'A' - 'a';
  }
  // GPUInfo("Key: key %d (%c) -> %d (%c) special %d (%c)", key, (char) key, (int) localeKey, localeKey, specialKey, (char) specialKey);

  if (specialKey) {
    keyOut = keyPressOut = specialKey;
  } else {
    keyOut = keyPressOut = localeKey;
    if (keyPressOut >= 'a' && keyPressOut <= 'z') {
      keyPressOut += 'A' - 'a';
    }
  }
}

void GPUDisplayBackendGlfw::error_callback(int error, const char* description) { fprintf(stderr, "Error: %s\n", description); }

void GPUDisplayBackendGlfw::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  int handleKey = 0, keyPress = 0;
  GetKey(key, scancode, mods, handleKey, keyPress);
  if (action == GLFW_PRESS) {
    me->mKeys[keyPress] = true;
    me->mKeysShift[keyPress] = mods & GLFW_MOD_SHIFT;
  } else if (action == GLFW_RELEASE) {
    if (me->mKeys[keyPress]) {
      me->HandleKeyRelease(handleKey);
    }
    me->mKeys[keyPress] = false;
    me->mKeysShift[keyPress] = false;
  }
}

void GPUDisplayBackendGlfw::mouseButton_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (action == GLFW_PRESS) {
    if (button == 0) {
      me->mMouseDn = true;
    } else if (button == 1) {
      me->mMouseDnR = true;
    }
    me->mMouseDnX = me->mouseMvX;
    me->mMouseDnY = me->mouseMvY;
  } else if (action == GLFW_RELEASE) {
    if (button == 0) {
      me->mMouseDn = false;
    } else if (button == 1) {
      me->mMouseDnR = false;
    }
  }
}

void GPUDisplayBackendGlfw::scroll_callback(GLFWwindow* window, double x, double y) { me->mMouseWheel += y * 100; }

void GPUDisplayBackendGlfw::cursorPos_callback(GLFWwindow* window, double x, double y)
{
  me->mouseMvX = x;
  me->mouseMvY = y;
}

void GPUDisplayBackendGlfw::resize_callback(GLFWwindow* window, int width, int height) { me->ReSizeGLScene(width, height); }

void GPUDisplayBackendGlfw::DisplayLoop()
{
#ifdef GPUCA_O2_LIB
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(me->mDisplayWidth, me->mDisplayHeight));
  ImGui::SetNextWindowBgAlpha(0.f);
  ImGui::Begin("Console", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
#endif
  me->DrawGLScene();
#ifdef GPUCA_O2_LIB
  ImGui::End();
#endif
}

int GPUDisplayBackendGlfw::OpenGLMain()
{
  me = this;

  if (!glfwInit()) {
    fprintf(stderr, "Error initializing glfw\n");
    return (-1);
  }
  glfwSetErrorCallback(error_callback);

  glfwWindowHint(GLFW_MAXIMIZED, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_MIN_VERSION_MAJOR);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_MIN_VERSION_MINOR);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 0);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GPUCA_DISPLAY_OPENGL_CORE_FLAGS ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);
  mWindow = glfwCreateWindow(INIT_WIDTH, INIT_HEIGHT, GL_WINDOW_NAME, nullptr, nullptr);
  if (!mWindow) {
    fprintf(stderr, "Error creating glfw window\n");
    glfwTerminate();
    return (-1);
  }
  glfwMakeContextCurrent(mWindow);

  glfwSetKeyCallback(mWindow, key_callback);
  glfwSetMouseButtonCallback(mWindow, mouseButton_callback);
  glfwSetScrollCallback(mWindow, scroll_callback);
  glfwSetCursorPosCallback(mWindow, cursorPos_callback);
  glfwSetWindowSizeCallback(mWindow, resize_callback);

  pthread_mutex_lock(&mSemLockExit);
  mGlfwRunning = true;
  pthread_mutex_unlock(&mSemLockExit);

  if (GPUDisplayExtInit()) {
    fprintf(stderr, "Error initializing GL extension wrapper\n");
    return (-1);
  }

#if defined(GPUCA_O2_LIB) && !defined(GPUCA_DISPLAY_GL3W)
  if (gl3wInit()) {
    fprintf(stderr, "Error initializing gl3w (2)\n");
    return (-1); // Hack: We have to initialize gl3w as well, as the DebugGUI uses it.
  }
#endif

  if (InitGL()) {
    fprintf(stderr, "Error in OpenGL initialization\n");
    return (1);
  }

#ifdef GPUCA_O2_LIB
  ImGui_ImplGlfwGL3_Init(mWindow, false);
  while (o2::framework::pollGUI(mWindow, DisplayLoop)) {
  }
#else
  while (!glfwWindowShouldClose(mWindow)) {
    HandleSendKey();
    if (DrawGLScene()) {
      fprintf(stderr, "Error drawing GL scene\n");
      return (1);
    }
    glfwSwapBuffers(mWindow);
    glfwPollEvents();
  }
#endif

  mDisplayControl = 2;
  pthread_mutex_lock(&mSemLockExit);
#ifdef GPUCA_O2_LIB
  ImGui_ImplGlfwGL3_Shutdown();
#endif
  glfwDestroyWindow(mWindow);
  glfwTerminate();
  mGlfwRunning = false;
  pthread_mutex_unlock(&mSemLockExit);

  return 0;
}

void GPUDisplayBackendGlfw::DisplayExit()
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

void GPUDisplayBackendGlfw::OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton)
{
#ifdef GPUCA_O2_LIB
  if (fromBotton) {
    y = ImGui::GetWindowHeight() - y;
  }
  y -= 20;
  ImGui::SetCursorPos(ImVec2(x, y));
  ImGui::TextColored(ImVec4(r, g, b, a), "%s", s);
#endif
}

void GPUDisplayBackendGlfw::SwitchFullscreen(bool set)
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

void GPUDisplayBackendGlfw::ToggleMaximized(bool set)
{
  if (set) {
    glfwMaximizeWindow(mWindow);
  } else {
    glfwRestoreWindow(mWindow);
  }
}

void GPUDisplayBackendGlfw::SetVSync(bool enable) { glfwSwapInterval(enable); }

int GPUDisplayBackendGlfw::StartDisplay()
{
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, OpenGLWrapper, this)) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
  return (0);
}

bool GPUDisplayBackendGlfw::EnableSendKey()
{
#ifdef GPUCA_O2_LIB
  return false;
#else
  return true;
#endif
}
