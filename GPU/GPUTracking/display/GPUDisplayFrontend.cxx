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

void* GPUDisplayFrontend::FrontendThreadWrapper(void* ptr)
{
  GPUDisplayFrontend* me = reinterpret_cast<GPUDisplayFrontend*>(ptr);
  int retVal = me->FrontendMain();
  if (retVal == -1) {
    me->InitDisplay(true);
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
int GPUDisplayFrontend::DrawGLScene() { return mDisplay->DrawGLScene(); }
void GPUDisplayFrontend::ResizeScene(int width, int height)
{
  mDisplayHeight = height;
  mDisplayWidth = width;
  mDisplay->ResizeScene(width, height);
}
int GPUDisplayFrontend::InitDisplay(bool initFailure) { return mDisplay->InitDisplay(initFailure); }
void GPUDisplayFrontend::ExitDisplay() { return mDisplay->ExitDisplay(); }
bool GPUDisplayFrontend::EnableSendKey() { return true; }

#ifdef _WIN32
#include "GPUDisplayFrontendWindows.h"
#elif defined(GPUCA_BUILD_EVENT_DISPLAY_X11)
#include "GPUDisplayFrontendX11.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLFW
#include "GPUDisplayFrontendGlfw.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLUT
#include "GPUDisplayFrontendGlut.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_WAYLAND
#include "GPUDisplayFrontendWayland.h"
#endif

GPUDisplayFrontend* GPUDisplayFrontend::getFrontend(const char* type)
{
#ifdef _WIN32
  if (strcmp(type, "windows") == 0) {
    return new GPUDisplayFrontendWindows;
  }
#elif defined(GPUCA_BUILD_EVENT_DISPLAY_X11)
  if (strcmp(type, "x11") == 0) {
    return new GPUDisplayFrontendX11;
  }
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLFW
  if (strcmp(type, "glfw") == 0) {
    return new GPUDisplayFrontendGlfw;
  }
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLUT
  if (strcmp(type, "glut") == 0) {
    return new GPUDisplayFrontendGlut;
  }
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_WAYLAND
  if (strcmp(type, "wayland") == 0) {
    return new GPUDisplayFrontendWayland;
  }
#endif
  return nullptr;
}

GPUDisplayBackend* GPUDisplayFrontend::backend()
{
  return mDisplay->backend();
}

int& GPUDisplayFrontend::drawTextFontSize()
{
  return mDisplay->drawTextFontSize();
}
