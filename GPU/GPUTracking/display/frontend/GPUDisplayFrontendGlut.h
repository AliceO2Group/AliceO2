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

/// \file GPUDisplayFrontendGlut.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDGLUT_H
#define GPUDISPLAYFRONTENDGLUT_H

#include "GPUDisplayFrontend.h"
#include <pthread.h>

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayFrontendGlut : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendGlut();
  ~GPUDisplayFrontendGlut() override = default;

  int32_t StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

 private:
  int32_t FrontendMain() override;

  static void displayFunc();
  static void glutLoopFunc();
  static void keyboardUpFunc(uint8_t key, int32_t x, int32_t y);
  static void keyboardDownFunc(uint8_t key, int32_t x, int32_t y);
  static void specialUpFunc(int32_t key, int32_t x, int32_t y);
  static void specialDownFunc(int32_t key, int32_t x, int32_t y);
  static void mouseMoveFunc(int32_t x, int32_t y);
  static void mMouseWheelFunc(int32_t button, int32_t dir, int32_t x, int32_t y);
  static void mouseFunc(int32_t button, int32_t state, int32_t x, int32_t y);
  static void ResizeSceneWrapper(int32_t width, int32_t height);
  static int32_t GetKey(int32_t key);
  static void GetKey(int32_t keyin, int32_t& keyOut, int32_t& keyPressOut, bool special);

  volatile bool mGlutRunning = false;
  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;

  int32_t mWidth = INIT_WIDTH;
  int32_t mHeight = INIT_HEIGHT;
  bool mFullScreen = false;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
