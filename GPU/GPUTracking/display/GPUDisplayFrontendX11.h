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

/// \file GPUDisplayFrontendX11.h
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDX11_H
#define GPUDISPLAYFRONTENDX11_H

#include "GPUDisplayFrontend.h"
#include <GL/glx.h>
#include <pthread.h>
#include <unistd.h>
#include <GL/glxext.h>

namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayFrontendX11 : public GPUDisplayFrontend
{
 public:
  GPUDisplayFrontendX11();
  ~GPUDisplayFrontendX11() override = default;

  int StartDisplay() override;
  void DisplayExit() override;
  void SwitchFullscreen(bool set) override;
  void ToggleMaximized(bool set) override;
  void SetVSync(bool enable) override;
  void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;
  void getSize(int& width, int& height) override;
  int getVulkanSurface(void* instance, void* surface) override;
  unsigned int getReqVulkanExtensions(const char**& p) override;

 private:
  int FrontendMain() override;
  int GetKey(int key);
  void GetKey(XEvent& event, int& keyOut, int& keyPressOut);

  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;
  volatile bool mDisplayRunning = false;

  GLuint mFontBase;

  Display* mDisplay = nullptr;
  Window mWindow;

  PFNGLXSWAPINTERVALEXTPROC mGlXSwapIntervalEXT = nullptr;
  bool vsync_supported = false;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
