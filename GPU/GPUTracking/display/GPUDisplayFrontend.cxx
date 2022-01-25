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

/// \file GPUDisplayFrontend.cxx
/// \author David Rohr

#include "GPUDisplayFrontend.h"
#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

void* GPUDisplayFrontend::OpenGLWrapper(void* ptr)
{
  GPUDisplayFrontend* me = reinterpret_cast<GPUDisplayFrontend*>(ptr);
  int retVal = me->OpenGLMain();
  if (retVal == -1) {
    me->InitGL(true);
  }
  return ((void*)(size_t)retVal);
}

void GPUDisplayFrontend::HandleSendKey()
{
  if (mSendKey) {
    mDisplay->HandleSendKey(mSendKey);
    mSendKey = 0;
  }
}

void GPUDisplayFrontend::HandleKey(unsigned char key) { mDisplay->HandleKey(key); }
int GPUDisplayFrontend::DrawGLScene(bool mixAnimation, float animateTime) { return mDisplay->DrawGLScene(mixAnimation, animateTime); }
void GPUDisplayFrontend::ReSizeGLScene(int width, int height)
{
  mDisplayHeight = height;
  mDisplayWidth = width;
  mDisplay->ReSizeGLScene(width, height);
}
int GPUDisplayFrontend::InitGL(bool initFailure) { return mDisplay->InitGL(initFailure); }
void GPUDisplayFrontend::ExitGL() { return mDisplay->ExitGL(); }
bool GPUDisplayFrontend::EnableSendKey() { return true; }

#ifdef GPUCA_BUILD_EVENT_DISPLAY
#ifdef _WIN32
#include "GPUDisplayFrontendWindows.h"
#else
#ifdef GPUCA_STANDALONE
#include "GPUDisplayFrontendX11.h"
#endif
#include "GPUDisplayFrontendGlfw.h"
#endif
#ifdef GPUCA_STANDALONE
#include "GPUDisplayFrontendGlut.h"
#endif
#endif

GPUDisplayFrontend* GPUDisplayFrontend::getFrontend(const char* type)
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY
#ifdef _WIN32
  if (strcmp(type, "windows") == 0) {
    return new GPUDisplayFrontendWindows;
  }
#else
#ifdef GPUCA_STANDALONE
  if (strcmp(type, "x11") == 0) {
    return new GPUDisplayFrontendX11;
  }
#endif
  if (strcmp(type, "glfw") == 0) {
    return new GPUDisplayFrontendGlfw;
  }
#endif
#ifdef GPUCA_STANDALONE
  if (strcmp(type, "glut") == 0) {
    return new GPUDisplayFrontendGlut;
  }
#endif
#endif
  return nullptr;
}
