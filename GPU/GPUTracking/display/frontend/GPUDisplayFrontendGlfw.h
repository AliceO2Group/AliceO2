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

  int32_t StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  bool EnableSendKey() override;
  void getSize(int32_t& width, int32_t& height) override;
  int32_t getVulkanSurface(void* instance, void* surface) override;
  uint32_t getReqVulkanExtensions(const char**& p) override;

 private:
  int32_t FrontendMain() override;
  static void DisplayLoop();

  static void error_callback(int32_t error, const char* description);
  static void key_callback(GLFWwindow* mWindow, int32_t key, int32_t scancode, int32_t action, int32_t mods);
  static void char_callback(GLFWwindow* window, uint32_t codepoint);
  static void mouseButton_callback(GLFWwindow* mWindow, int32_t button, int32_t action, int32_t mods);
  static void scroll_callback(GLFWwindow* mWindow, double x, double y);
  static void cursorPos_callback(GLFWwindow* mWindow, double x, double y);
  static void resize_callback(GLFWwindow* mWindow, int32_t width, int32_t height);
  static int32_t GetKey(int32_t key);
  static void GetKey(int32_t keyin, int32_t scancode, int32_t mods, int32_t& keyOut, int32_t& keyPressOut);

  GLFWwindow* mWindow;

  volatile bool mGlfwRunning = false;
  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;
  int32_t mWindowX = 0;
  int32_t mWindowY = 0;
  int32_t mWindowWidth = INIT_WIDTH;
  int32_t mWindowHeight = INIT_HEIGHT;
  uint8_t mKeyDownMap[256] = {0};
  uint8_t mLastKeyDown = 0;
  bool mUseIMGui = false;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
