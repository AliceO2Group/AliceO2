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

#include <QApplication>
#include <QTimer>

#include <thread>
#include <mutex>
#include <condition_variable>

using namespace GPUCA_NAMESPACE::gpu;

namespace GPUCA_NAMESPACE::gpu
{
struct GPUDisplayGUIWrapperObjects {
  std::unique_ptr<QApplication> app;
  std::unique_ptr<GPUDisplayGUI> gui;
  std::unique_ptr<QTimer> timer;
  std::thread t;
  volatile bool start = false;
  volatile bool stop = false;
  volatile bool terminate = false;
  volatile bool started = false;
  volatile bool stopped = false;
  std::mutex mutex, mutexRet;
  std::condition_variable signal, signalRet;
};
} // namespace GPUCA_NAMESPACE::gpu

GPUDisplayGUIWrapper::GPUDisplayGUIWrapper()
{
  static bool first = true;
  static std::mutex mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    if (!first) {
      throw std::runtime_error("Must not start QApplication twice");
      first = true;
    }
  }
  mO.reset(new GPUDisplayGUIWrapperObjects);
  mO->t = std::thread(&GPUDisplayGUIWrapper::guiThread, this);
}
GPUDisplayGUIWrapper::~GPUDisplayGUIWrapper()
{
  stop();
  {
    std::lock_guard<std::mutex> guard(mO->mutex);
    mO->terminate = true;
    mO->signal.notify_one();
  }
  mO->t.join();
}

void GPUDisplayGUIWrapper::UpdateTimer()
{
  if (mO->stop) {
    mO->gui->close();
    mO->stop = false;
  }
}

void GPUDisplayGUIWrapper::guiThread()
{
  static int tmp_argc = 1;
  static const char* tmp_argv[2] = {"GPU CA Standalone Event Display GUI", NULL};
  mO->app.reset(new QApplication(tmp_argc, (char**)tmp_argv));
  while (!mO->terminate) {
    {
      std::unique_lock<std::mutex> lock(mO->mutex);
      if (!mO->start && !mO->terminate) {
        mO->signal.wait(lock);
      }
    }
    if (mO->terminate) {
      break;
    }
    if (mO->start) {
      mO->start = false;
      mO->stopped = false;
      mO->gui.reset(new GPUDisplayGUI);
      mO->gui->setWrapper(this);
      mO->gui->show();
      mO->timer.reset(new QTimer(mO->gui.get()));
      mO->timer->start(10);
      mO->gui->connect(mO->timer.get(), SIGNAL(timeout()), mO->gui.get(), SLOT(UpdateTimer()));
      {
        std::lock_guard<std::mutex> guard(mO->mutexRet);
        mO->started = true;
        mO->signalRet.notify_one();
      }
      mO->app->exec();
      mO->timer->stop();
      mO->timer.reset(nullptr);
      mO->gui.reset(nullptr);
      std::lock_guard<std::mutex> guard(mO->mutexRet);
      mO->started = false;
      mO->stopped = true;
      mO->signalRet.notify_one();
    }
  }
  mO->app.reset(nullptr);
}

int GPUDisplayGUIWrapper::start()
{
  if (!mO->started) {
    {
      std::lock_guard<std::mutex> guard(mO->mutex);
      mO->start = true;
      mO->signal.notify_one();
    }
    {
      std::unique_lock<std::mutex> lock(mO->mutexRet);
      while (mO->started == false) {
        mO->signalRet.wait(lock);
      }
    }
  }
  return 0;
}

int GPUDisplayGUIWrapper::stop()
{
  if (mO->started) {
    mO->stop = true;
    {
      std::unique_lock<std::mutex> lock(mO->mutexRet);
      while (mO->stopped == false) {
        mO->signalRet.wait(lock);
      }
    }
  }
  return 0;
}

int GPUDisplayGUIWrapper::focus()
{
  if (mO->started) {
  }
  return 0;
}

bool GPUDisplayGUIWrapper::isRunning() const
{
  return mO->started;
}
