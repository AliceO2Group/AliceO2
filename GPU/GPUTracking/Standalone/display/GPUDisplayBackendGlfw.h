// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendGlfw.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDGlfw_H
#define GPUDISPLAYBACKENDGlfw_H

#include "GPUDisplayBackend.h"
#include <pthread.h>

struct GLFWwindow;

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayBackendGlfw : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendGlfw() = default;
  ~GPUDisplayBackendGlfw() override = default;

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  bool EnableSendKey() override;

 private:
  int OpenGLMain() override;
  static void DisplayLoop();

  static void error_callback(int error, const char* description);
  static void key_callback(GLFWwindow* mWindow, int key, int scancode, int action, int mods);
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
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
