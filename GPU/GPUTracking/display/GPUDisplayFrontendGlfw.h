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

/// \file GPUDisplayFrontendGlfw.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDGLFW_H
#define GPUDISPLAYFRONTENDGLFW_H

#include "GPUDisplayFrontend.h"
#include <pthread.h>

struct GLFWwindow;

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayFrontendGlfw : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendGlfw();
  ~GPUDisplayFrontendGlfw() override = default;

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  bool EnableSendKey() override;
  void getSize(int& width, int& height) override;
  int getVulkanSurface(void* instance, void* surface) override;
  unsigned int getReqVulkanExtensions(const char**& p) override;

 private:
  int FrontendMain() override;
  static void DisplayLoop();

  static void error_callback(int error, const char* description);
  static void key_callback(GLFWwindow* mWindow, int key, int scancode, int action, int mods);
  static void char_callback(GLFWwindow* window, unsigned int codepoint);
  static void mouseButton_callback(GLFWwindow* mWindow, int button, int action, int mods);
  static void scroll_callback(GLFWwindow* mWindow, double x, double y);
  static void cursorPos_callback(GLFWwindow* mWindow, double x, double y);
  static void resize_callback(GLFWwindow* mWindow, int width, int height);
  static int GetKey(int key);
  static void GetKey(int keyin, int scancode, int mods, int& keyOut, int& keyPressOut);

  GLFWwindow* mWindow;

  volatile bool mGlfwRunning = false;
  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;
  int mWindowX = 0;
  int mWindowY = 0;
  int mWindowWidth = INIT_WIDTH;
  int mWindowHeight = INIT_HEIGHT;
  char mKeyDownMap[256] = {0};
  unsigned char mLastKeyDown = 0;
  bool mUseIMGui = false;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
