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
#include "GPUDisplayConsole.h"
#include "GPUDisplay.h"

#ifdef _WIN32
#include "GPUDisplayFrontendWindows.h"
#endif
#include <system_error>
#include <thread>

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

#include "GPUDisplayGUIChannel.h"
#include "GPULogging.h"
#include <cstring>
#include <exception>
#include <algorithm>
#include <iterator>
#include <utility>

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

GPUDisplayFrontend::GPUDisplayFrontend() : mConsole(std::make_unique<GPUDisplayConsole>(std::string(DISPLAY_WINDOW_NAME) + " Console")) {}
GPUDisplayFrontend::~GPUDisplayFrontend()
{
  if (mFrontendThread.joinable()) { // Started but not stopped, as the detached thread before
    mFrontendThread.detach();
  }
}

void GPUDisplayFrontend::FrontendThreadWrapper()
{
  int32_t retVal;
  try {
    retVal = FrontendMain();
  } catch (const std::exception& e) {
    GPUError("Error in display frontend: %s", e.what());
    retVal = -1;
  }
  if (retVal) {
    std::unique_lock lock(mExitStateMutex);
    if (mDisplay->mInitResult == 0) { // Failed before InitDisplay was called, report failure to StartDisplay(), which otherwise waits forever
      lock.unlock();
      InitDisplay(true);
    } else {
      mDisplay->mInitResult = -1; // Error after the initialization: StopDisplay does not reset the state for a new start
    }
  }
}

void GPUDisplayFrontend::waitForFPSLimit()
{
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
  std::unique_lock lock(mConfigUpdateMutex);
  mConfigUpdateCV.wait_until(lock, mNextFrameTime, [this] { return mConfigUpdatePending; });
}

void GPUDisplayFrontend::signalConfigUpdate()
{
  {
    std::lock_guard lock(mConfigUpdateMutex);
    mConfigUpdatePending = true;
  }
  mConfigUpdateCV.notify_one();
}

void GPUDisplayFrontend::HandleExternalGUISignals()
{
  {
    std::lock_guard lock(mConfigUpdateMutex);
    mConfigUpdatePending = false; // Before fetching the updates, so that updates sent in the meantime signal again
  }
  HandleExternalGUIKeys();

  const bool gui = isGUIRunning();
  std::optional<GPUDisplayGUIChannel::Update> update;
  if (gui) {
    std::lock_guard lock(mGUIChannel->mutex);
    update = std::exchange(mGUIChannel->update, std::nullopt);
    mGUIChannel->displayProcessingCurrentUpdate = update && mGUIChannel->guiSleeping; // GUI waits for the reply when it wakes up
  }
  if (update) {
    for (int32_t key : update->keys) {
      mDisplay->HandleExternalGUIKey(key);
    }
    applyGUIValues(*update);
    if (update->guiScale) {
      mDisplay->mCfgC.mGuiScale = *update->guiScale;
    }
    if (update->guiDarkMode) {
      mDisplay->mCfgC.mGuiDarkMode = *update->guiDarkMode;
    }
  }

  if (gui) {
    GPUDisplayGUIChannel::Settings cfg{mDisplay->mCfgH, mDisplay->mCfgL, mDisplay->mCfgR, mDisplay->mCfgC};
    {
      std::lock_guard lock(mGUIChannel->mutex);
      mGUIChannel->state.cfg = std::move(cfg);
      mGUIChannel->state.infoStatus = mDisplay->mCfgC.mPrintInfoTextGUI ? mDisplay->mGUIInfoStatus : "";
      mGUIChannel->state.infoText = mDisplay->mCfgC.mPrintInfoTextGUI ? mDisplay->mGUIInfoText : "";
      mGUIChannel->state.infoTextCounter = mDisplay->mGUIInfoTextCounter;
      if (mDisplay->mUpdateGUIOptions) {
        mGUIChannel->state.optionsRevision++;
        mDisplay->mUpdateGUIOptions = false;
      }
      if (update) {
        mGUIChannel->state.appliedUpdate = update->id;
      }
      mGUIChannel->state.counter++;
      mGUIChannel->displayProcessingCurrentUpdate = false;
    }
    mGUIChannel->stateUpdated.notify_all();
    std::unique_lock lock(mGUIChannel->mutex);
    if (mGUIChannel->optionsReadRevision != mGUIChannel->state.optionsRevision) { // The GUI calls the dynamic option lambdas (reading display data) only while we wait here, short timeout, retried in the next frame
      mGUIChannel->displayWaitingForOptions = true;
      mGUIChannel->optionsRead.wait_for(lock, std::chrono::milliseconds(20), [this] { return mGUIChannel->optionsReadRevision == mGUIChannel->state.optionsRevision || mGUIChannel->stop; });
      mGUIChannel->displayWaitingForOptions = false;
    }
  }
}

void GPUDisplayFrontend::sendExternalGUIKey(int32_t v, uint32_t translationMode, uint64_t extendedKey)
{
  if (translationMode && (v = translateExternalKey(translationMode, extendedKey)) < 0) {
    return;
  }
  {
    std::lock_guard lock(mExternalGUIKeyMutex);
    mExternalGUIKeys.emplace_back(v);
  }
  signalConfigUpdate();
}

bool GPUDisplayFrontend::consoleShown() const { return mDisplay->mCfgC.showConsole; }

int32_t GPUDisplayFrontend::getDisplayControl() const { return mDisplay->mCfgC.mDisplayControl.load(); }
void GPUDisplayFrontend::setDisplayControl(int32_t v) { mDisplay->mCfgC.mDisplayControl.store(v); }

void GPUDisplayFrontend::HandleKey(uint8_t key) { mDisplay->HandleKey(key); }
int32_t GPUDisplayFrontend::HandleUpdatesAndDrawGLScene(bool suspended)
{
  HandleExternalGUISignals();
  mDisplay->checkAndApplyConfigChanges();
  const int32_t retVal = mDisplay->DrawGLScene(suspended);
  mDisplay->mOldCfg = {mDisplay->mCfgH, mDisplay->mCfgL, mDisplay->mCfgR}; // Also includes the changes during drawing (animation, scene reset, ...), which set their update flags themselves
  return retVal;
}

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
    mGUIChannel = std::make_unique<GPUDisplayGUIChannel>();
    mGUIChannel->updatePosted = [this]() { signalConfigUpdate(); };
    mGUI = std::make_unique<GPUDisplayGUIWrapper>(*mGUIChannel);
  }
  if (!mGUI->isRunning()) {
    {
      std::lock_guard lock(mGUIChannel->mutex);
      mGUIChannel->update.reset();
      mGUIChannel->state = {};
      mGUIChannel->guiSleeping = mGUIChannel->displayProcessingCurrentUpdate = mGUIChannel->stop = mGUIChannel->displayWaitingForOptions = false;
      mGUIChannel->optionsReadRevision = 0;
      mDisplay->mUpdateGUIOptions = true;
      mGUIChannel->replyTimeout = std::chrono::milliseconds(mDisplay->cfg().guiReplyTimeout);
      mGUIChannel->maxUpdatesAhead = std::max<uint32_t>(mDisplay->cfg().guiMaxUpdatesAhead, 1); // 0 would block all updates
      mGUIChannel->windowName = DISPLAY_WINDOW_NAME;
      setGUILayout(*mGUIChannel);
    }
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
  mDisplay->mCfgR.fullScreen = mWindowFullScreen = fullScreen;
  mDisplay->mCfgR.maximized = mWindowMaximized = maximized;
}

void GPUDisplayFrontend::requestFullScreen(bool set)
{
  if (set != std::exchange(mWindowFullScreen, set)) {
    SwitchFullscreen(set);
  }
}

void GPUDisplayFrontend::requestMaximized(bool set)
{
  if (set != std::exchange(mWindowMaximized, set)) {
    ToggleMaximized(set);
  }
}

void GPUDisplayFrontend::focusLost()
{
  std::fill(std::begin(mKeys), std::end(mKeys), false);
  mMouseDn = mMouseDnR = mMouseDnM = false;
}

void GPUDisplayFrontend::applyInitialWindowState()
{
  if (mDisplay->cfgR().fullScreen) {
    requestFullScreen(true);
  } else if (mDisplay->cfgR().maximized) {
    requestMaximized(true);
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
  try {
    mFrontendThread = std::thread(&GPUDisplayFrontend::FrontendThreadWrapper, this);
  } catch (const std::system_error& e) {
    GPUError("Could not create frontend thread: %s", e.what());
    return (1);
  }
  std::unique_lock lock(mExitStateMutex);
  mStartDisplayDone = true;
  mInitCV.notify_all();
  mInitCV.wait(lock, [this] { return mDisplay->mInitResult != 0; }); // Both threads have finished their initialization
  return (mDisplay->mInitResult != 1);
}

void GPUDisplayFrontend::StopDisplay()
{
  {
    std::unique_lock lock(mExitStateMutex);
    if (mDisplayRunning) {
      mDisplay->mCfgC.mDisplayControl.store(2);
      StopDisplayA();
      mDisplay->UnblockDisplay();
    }
    mDisplayStoppedCV.wait(lock, [this] { return !mDisplayRunning; });
  }
  if (mFrontendThread.joinable()) { // Also the wrapper has finished (it sets mInitResult = -1 for errors after the initialization), not holding mExitStateMutex, which the wrapper uses
    mFrontendThread.join();
  }
  std::lock_guard lock(mExitStateMutex);
  if (mDisplay->mInitResult == 1) { // For a new StartDisplay, not after a failed initialization or an error (undefined state)
    mStartDisplayDone = false;
    mDisplay->mInitResult = 0;
  }
}

void GPUDisplayFrontend::StopDisplayA() {}
