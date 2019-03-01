// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendX11.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDX11_H
#define GPUDISPLAYBACKENDX11_H

#include "GPUDisplayBackend.h"
#include <GL/glx.h>
#include <pthread.h>
#include <unistd.h>
#include <GL/glxext.h>

class GPUDisplayBackendX11 : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendX11() = default;
  virtual ~GPUDisplayBackendX11() = default;

  virtual int StartDisplay() override;
  virtual void DisplayExit() override;
  virtual void SwitchFullscreen(bool set) override;
  virtual void ToggleMaximized(bool set) override;
  virtual void SetVSync(bool enable) override;
  virtual void OpenGLPrint(const char* s, float x, float y, float r, float g, float b, float a, bool fromBotton = true) override;

 private:
  virtual int OpenGLMain();
  int GetKey(int key);
  void GetKey(XEvent& event, int& keyOut, int& keyPressOut);

  pthread_mutex_t mSemLockExit = PTHREAD_MUTEX_INITIALIZER;
  volatile bool mDisplayRunning = false;

  GLuint mFontBase;

  Display* mDisplay = NULL;
  Window mWindow;

  PFNGLXSWAPINTERVALEXTPROC mGlXSwapIntervalEXT = NULL;
};

#endif
