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

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

 private:
  int FrontendMain() override;

  static void displayFunc();
  static void glutLoopFunc();
  static void keyboardUpFunc(unsigned char key, int x, int y);
  static void keyboardDownFunc(unsigned char key, int x, int y);
  static void specialUpFunc(int key, int x, int y);
  static void specialDownFunc(int key, int x, int y);
  static void mouseMoveFunc(int x, int y);
  static void mMouseWheelFunc(int button, int dir, int x, int y);
  static void mouseFunc(int button, int state, int x, int y);
  static void ResizeSceneWrapper(int width, int height);
  static int GetKey(int key);
  static void GetKey(int keyin, int& keyOut, int& keyPressOut, bool special);

  volatile bool mGlutRunning = false;
  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;

  int mWidth = INIT_WIDTH;
  int mHeight = INIT_HEIGHT;
  bool mFullScreen = false;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
