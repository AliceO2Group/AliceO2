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

/// \file GPUDisplayGUIWrapper.cxx
/// \author David Rohr

#include "GPUDisplayGUIWrapper.h"
#include "GPUDisplayGUI.h"
#include "GPUDisplayGUIChannel.h"

#include <QApplication>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace o2::gpu;

namespace o2::gpu::internal
{
struct GPUDisplayGUIWrapperObjects {
  int argc = 1;
  char appName[64] = "GPU CA Standalone Event Display GUI";
  char* argv[2] = {appName, nullptr}; // Must outlive the QApplication
  std::unique_ptr<QApplication> app;
  std::unique_ptr<GPUDisplayGUI> gui; // Protected by mutex
  std::thread thread;
  std::mutex mutex;
  std::condition_variable signal;
  bool start = false, terminate = false; // Protected by mutex
  std::atomic<bool> started = false;     // Written under mutex
};
} // namespace o2::gpu::internal

namespace
{
std::atomic<bool> instanceExists = false; // Qt supports only one QApplication
} // namespace

GPUDisplayGUIWrapper::GPUDisplayGUIWrapper(GPUDisplayGUIChannel& channel) : mChannel(channel)
{
  if (instanceExists.exchange(true)) {
    throw std::runtime_error("Must not start QApplication twice");
  }
  mO = std::make_unique<internal::GPUDisplayGUIWrapperObjects>();
  mO->thread = std::thread(&GPUDisplayGUIWrapper::guiThread, this);
}

GPUDisplayGUIWrapper::~GPUDisplayGUIWrapper()
{
  stop();
  {
    std::lock_guard lock(mO->mutex);
    mO->terminate = true;
  }
  mO->signal.notify_all();
  mO->thread.join();
  instanceExists = false;
}

void GPUDisplayGUIWrapper::guiThread()
{
  mO->app = std::make_unique<QApplication>(mO->argc, mO->argv);
  std::unique_lock lock(mO->mutex);
  while (true) {
    mO->signal.wait(lock, [this] { return mO->start || mO->terminate; });
    if (mO->terminate) {
      break;
    }
    mO->start = false;
    mO->gui = std::make_unique<GPUDisplayGUI>(mChannel);
    mO->gui->show();
    mO->started = true;
    mO->signal.notify_all();
    lock.unlock();
    mO->app->exec(); // Returns when the GUI window is closed, by the user or by stop()
    lock.lock();
    mO->gui.reset();
    mO->started = false;
    mO->signal.notify_all();
  }
  lock.unlock();
  mO->app.reset();
}

int32_t GPUDisplayGUIWrapper::start()
{
  std::unique_lock lock(mO->mutex);
  if (!mO->started) {
    mO->start = true;
    mO->signal.notify_all();
    mO->signal.wait(lock, [this] { return mO->started.load(); });
  }
  return 0;
}

int32_t GPUDisplayGUIWrapper::stop()
{
  {
    std::lock_guard lock(mChannel.mutex);
    mChannel.stop = true;
  }
  mChannel.stateUpdated.notify_all(); // Wake up the GUI if it waits for a reply
  std::unique_lock lock(mO->mutex);
  mO->signal.wait(lock, [this] { return !mO->started; });
  return 0;
}

int32_t GPUDisplayGUIWrapper::focus()
{
  std::lock_guard lock(mO->mutex);
  if (mO->started) {
    QMetaObject::invokeMethod(mO->gui.get(), &GPUDisplayGUI::bringToFront, Qt::QueuedConnection); // Widgets must only be accessed from the GUI thread
  }
  return 0;
}

bool GPUDisplayGUIWrapper::isRunning() const
{
  return mO->started;
}
