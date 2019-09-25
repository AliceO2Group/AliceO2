// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendGlut.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDGLUT_H
#define GPUDISPLAYBACKENDGLUT_H

#include "GPUDisplayBackend.h"
#include <pthread.h>

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayBackendGlut : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendGlut() = default;
  ~GPUDisplayBackendGlut() override = default;

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

 private:
  int OpenGLMain() override;

  static void displayFunc(void);
  static void glutLoopFunc(void);
  static void keyboardUpFunc(unsigned char key, int x, int y);
  static void keyboardDownFunc(unsigned char key, int x, int y);
  static void specialUpFunc(int key, int x, int y);
  static void specialDownFunc(int key, int x, int y);
  static void mouseMoveFunc(int x, int y);
  static void mMouseWheelFunc(int button, int dir, int x, int y);
  static void mouseFunc(int button, int state, int x, int y);
  static void ReSizeGLSceneWrapper(int width, int height);
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
