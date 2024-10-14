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

#ifdef GPUCA_BUILD_EVENT_DISPLAY_QT
#include "GPUDisplayGUIWrapper.h"
#else
namespace GPUCA_NAMESPACE::gpu
{
class GPUDisplayGUIWrapper
{
};
} // namespace GPUCA_NAMESPACE::gpu
#endif

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayFrontend::~GPUDisplayFrontend() = default;

void* GPUDisplayFrontend::FrontendThreadWrapper(void* ptr)
{
  GPUDisplayFrontend* me = reinterpret_cast<GPUDisplayFrontend*>(ptr);
  int32_t retVal = me->FrontendMain();
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

void GPUDisplayFrontend::HandleKey(uint8_t key) { mDisplay->HandleKey(key); }
int32_t GPUDisplayFrontend::DrawGLScene() { return mDisplay->DrawGLScene(); }
void GPUDisplayFrontend::ResizeScene(int32_t width, int32_t height)
{
  mDisplayHeight = height;
  mDisplayWidth = width;
  mDisplay->ResizeScene(width, height);
}
int32_t GPUDisplayFrontend::InitDisplay(bool initFailure) { return mDisplay->InitDisplay(initFailure); }
void GPUDisplayFrontend::ExitDisplay()
{
  mDisplay->ExitDisplay();
  stopGUI();
  mGUI.reset(nullptr);
}
bool GPUDisplayFrontend::EnableSendKey() { return true; }

void GPUDisplayFrontend::stopGUI()
{
#ifdef GPUCA_BUILD_EVENT_DISPLAY_QT
  if (mGUI) {
    mGUI->stop();
  }
#endif
}

int32_t GPUDisplayFrontend::startGUI()
{
  int32_t retVal = 1;
#ifdef GPUCA_BUILD_EVENT_DISPLAY_QT
  if (!mGUI) {
    mGUI.reset(new GPUDisplayGUIWrapper);
  }
  if (!mGUI->isRunning()) {
    mGUI->start();
  } else {
    mGUI->focus();
  }
#endif
  return retVal;
}

bool GPUDisplayFrontend::isGUIRunning()
{
  bool retVal = false;
#ifdef GPUCA_BUILD_EVENT_DISPLAY_QT
  retVal = mGUI && mGUI->isRunning();
#endif
  return retVal;
}

GPUDisplayFrontend* GPUDisplayFrontend::getFrontend(const char* type)
{
#if !defined(GPUCA_STANDALONE) && defined(GPUCA_BUILD_EVENT_DISPLAY_GLFW)
  if (strcmp(type, "glfw") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendGlfw;
  } else
#endif
#ifdef _WIN32
  if (strcmp(type, "windows") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendWindows;
  } else
#elif defined(GPUCA_BUILD_EVENT_DISPLAY_X11)
  if (strcmp(type, "x11") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendX11;
  } else
#endif
#if defined(GPUCA_STANDALONE) && defined(GPUCA_BUILD_EVENT_DISPLAY_GLFW)
  if (strcmp(type, "glfw") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendGlfw;
  } else
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_WAYLAND
  if (strcmp(type, "wayland") == 0 || (strcmp(type, "auto") == 0 && getenv("XDG_SESSION_TYPE") && strcmp(getenv("XDG_SESSION_TYPE"), "wayland") == 0)) {
    return new GPUDisplayFrontendWayland;
  } else
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLUT
  if (strcmp(type, "glut") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendGlut;
  } else
#endif
  {
    GPUError("Requested frontend not available");
  }
  return nullptr;
}

GPUDisplayBackend* GPUDisplayFrontend::backend()
{
  return mDisplay->backend();
}

int32_t& GPUDisplayFrontend::drawTextFontSize()
{
  return mDisplay->drawTextFontSize();
}
