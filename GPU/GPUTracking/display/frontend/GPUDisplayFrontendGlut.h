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

namespace o2::gpu
{
class GPUDisplayFrontendGlut : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendGlut();
  ~GPUDisplayFrontendGlut() override = default;

  void StopDisplayA() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  bool canSetVSync() override;

 private:
  int32_t FrontendMain() override;

  static GPUDisplayFrontendGlut* self(void* p) { return static_cast<GPUDisplayFrontendGlut*>(p); } // GLUT callback user data
  void displayFunc();
  void glutLoopFunc();
  void keyboardFunc(int32_t key, bool special, bool down);
  void mouseMoveFunc(int32_t x, int32_t y);
  void mMouseWheelFunc(int32_t button, int32_t dir, int32_t x, int32_t y);
  void mouseFunc(int32_t button, int32_t state, int32_t x, int32_t y);
  void ResizeSceneWrapper(int32_t width, int32_t height);
  static int32_t GetKey(int32_t key);
  static void GetKey(int32_t keyin, int32_t& keyOut, int32_t& keyPressOut, bool special);

  int32_t mWidth = INIT_WIDTH;
  int32_t mHeight = INIT_HEIGHT;
  bool mFullScreen = false;
  bool mRedisplayPending = false; // glutPostRedisplay called in glutLoopFunc, not yet served by displayFunc
};
} // namespace o2::gpu

#endif
