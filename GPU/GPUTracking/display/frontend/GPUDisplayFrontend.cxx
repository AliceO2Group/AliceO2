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
#else
#include <pthread.h>
#endif

#ifdef GPUCA_BUILD_EVENT_DISPLAY_X11
#include "GPUDisplayFrontendX11.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLFW
#include "GPUDisplayFrontendGlfw.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLUT
#include "GPUDisplayFrontendGlut.h"
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_WAYLAND
namespace o2::gpu
{
GPUDisplayFrontend* createGPUDisplayFrontendWayland();
} // namespace o2::gpu
#endif
#include "GPUDisplayFrontendNone.h"

#include "GPULogging.h"
#include <cstring>
#include <exception>
#include <algorithm>
#include <iterator>
#include <thread>

#ifdef GPUCA_BUILD_EVENT_DISPLAY_QT
#include "GPUDisplayGUIWrapper.h"
#else
namespace o2::gpu
{
class GPUDisplayGUIWrapper
{
};
} // namespace o2::gpu
#endif

using namespace o2::gpu;

GPUDisplayFrontend::~GPUDisplayFrontend() = default;

void* GPUDisplayFrontend::FrontendThreadWrapper(void* ptr)
{
  GPUDisplayFrontend* me = reinterpret_cast<GPUDisplayFrontend*>(ptr);
  int32_t retVal;
  try {
    retVal = me->FrontendMain();
  } catch (const std::exception& e) {
    GPUError("Error in display frontend: %s", e.what());
    retVal = -1;
  }
  if (retVal && me->mDisplay->mInitResult == 0) { // Failed before InitDisplay was called, report failure to StartDisplay(), which otherwise waits forever
    me->InitDisplay(true);
  }
  return ((void*)(size_t)retVal);
}

void GPUDisplayFrontend::waitForFPSLimit()
{
  mNeedUpdate = 0;
  const auto now = std::chrono::steady_clock::now();
  const auto& cfg = mDisplay->cfgR();
  if (mBackend->fpsLimitMode() != GPUDisplayBackend::FPS_LIMIT_RATE || cfg.fpsRate <= 0) {
    mNextFrameTime = now;
    return;
  }
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1. / cfg.fpsRate));
  mNextFrameTime += period;
  if (mNextFrameTime < now - period) { // More than one frame behind (slow rendering, first frame), do not try to catch up
    mNextFrameTime = now;
  }
  std::this_thread::sleep_until(mNextFrameTime);
}

void GPUDisplayFrontend::HandleSendKeys()
{
  std::lock_guard lock(mSendKeyMutex);
  for (auto key : mSendKeys) {
    mDisplay->HandleSendKey(key);
  }
  mSendKeys.clear();
}

void GPUDisplayFrontend::sendKey(int32_t v)
{
  std::lock_guard lock(mSendKeyMutex);
  mSendKeys.emplace_back(v);
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
{ // clang-format off
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
    return createGPUDisplayFrontendWayland();
  } else
#endif
#ifdef GPUCA_BUILD_EVENT_DISPLAY_GLUT
  if (strcmp(type, "glut") == 0 || strcmp(type, "auto") == 0) {
    return new GPUDisplayFrontendGlut;
  } else
#endif
  if (strcmp(type, "none") == 0) {
    return new GPUDisplayFrontendNone;
  } else {
    GPUError("Requested frontend not available");
  }
  return nullptr;
} // clang-format on

GPUDisplayBackend* GPUDisplayFrontend::backend()
{
  return mDisplay->backend();
}

void GPUDisplayFrontend::setWindowState(bool fullScreen, bool maximized)
{
  mDisplay->mCfgR.fullScreen = fullScreen;
  mDisplay->mCfgR.maximized = maximized;
}

void GPUDisplayFrontend::focusLost()
{
  std::fill(std::begin(mKeys), std::end(mKeys), false);
  std::fill(std::begin(mKeysShift), std::end(mKeysShift), false);
  mMouseDn = mMouseDnR = mMouseDnM = false;
}

void GPUDisplayFrontend::applyInitialWindowState()
{
  if (mDisplay->cfgR().fullScreen) {
    SwitchFullscreen(true);
  } else if (mDisplay->cfgR().maximized) {
    ToggleMaximized(true);
  }
}

bool GPUDisplayFrontend::vsyncEnabled() const
{
  return mDisplay->cfgR().drawQualityVSync;
}

int32_t& GPUDisplayFrontend::drawTextFontSize()
{
  return mDisplay->drawTextFontSize();
}

int32_t GPUDisplayFrontend::StartDisplay()
{
#ifndef _WIN32
  static pthread_t hThread;
  if (pthread_create(&hThread, nullptr, FrontendThreadWrapper, this)) {
    GPUError("Coult not Create frontend Thread...");
    return (1);
  }
#else
  HANDLE hThread;
  if ((hThread = CreateThread(nullptr, nullptr, &OpenGLWrapper, this, nullptr, nullptr)) == nullptr) {
    GPUError("Coult not Create GL Thread...");
    return (1);
  }
#endif
  return (0);
}

void GPUDisplayFrontend::StopDisplay()
{
  {
    std::lock_guard lock(mExitStateMutex);
    if (mDisplayRunning) {
      mDisplayControl = 2;
      StopDisplayA();
      mDisplay->UnblockDisplay();
    }
  }
  while (mDisplayRunning) {
    std::this_thread::sleep_for(std::chrono::microseconds(10000)); // TODO: Signal by semaphore
  }
}

void GPUDisplayFrontend::StopDisplayA() {}
