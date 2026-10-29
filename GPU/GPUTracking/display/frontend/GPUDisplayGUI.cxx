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

/// \file GPUDisplayGUI.cxx
/// \author David Rohr

#include "GPUDisplayGUI.h"
#include "GPUDisplayFrontend.h"
#include "ui_GPUDisplayGUI.h"

#include <QKeyEvent>
#include <QSignalBlocker>

#include <chrono>
#include <cmath>

using namespace o2::gpu;

GPUDisplayGUI::GPUDisplayGUI(GPUDisplayGUIChannel& channel, QWidget* parent) : QMainWindow(parent), mUi(std::make_unique<Ui::GPUDisplayGUI>()), mChannel(channel)
{
  mUi->setupUi(this);
  {
    std::lock_guard lock(mChannel.mutex);
    setWindowTitle(QString::fromStdString(mChannel.windowName + " GUI"));
  }
  connect(mUi->lineWidthSlider, &QSlider::valueChanged, this, [this](int32_t value) {
    mUpdate.lineWidth = value * 0.1f;
    mUi->lineWidthValue->setNum(*mUpdate.lineWidth);
  });
  connect(&mTimer, &QTimer::timeout, this, &GPUDisplayGUI::timerTick);
  mTimer.start(std::chrono::milliseconds(10));
}

GPUDisplayGUI::~GPUDisplayGUI() = default;

void GPUDisplayGUI::timerTick()
{
  std::unique_lock lock(mChannel.mutex);
  mChannel.guiSleeping = false;
  if (mUpdate.empty()) { // Display processes the update we went to sleep with, wait for the reply (with new changes we would not apply it anyway)
    mChannel.stateUpdated.wait_for(lock, mChannel.replyTimeout, [this] { return !mChannel.displayProcessingCurrentUpdate || mChannel.stop; });
  }
  if (!mUpdate.empty() && mLastSentId - mLastAppliedId >= mChannel.maxUpdatesAhead) { // Too far ahead of the display, wait for the reply to the last update
    mChannel.stateUpdated.wait_for(lock, mChannel.replyTimeout, [this] { return mChannel.state.appliedUpdate == mLastSentId || mChannel.stop; });
  }
  if (mChannel.stop) {
    lock.unlock();
    close();
    return;
  }

  const bool newState = mChannel.state.counter != mState.counter;
  if (newState) {
    mState = mChannel.state;
    if (mState.appliedUpdate == mLastSentId) {
      mLastAppliedId = mLastSentId;
    }
  }
  const bool apply = newState && mUpdate.empty() && mState.appliedUpdate == mLastSentId; // State contains all our updates, and we have no unsent changes
  const bool post = !mUpdate.empty() && mLastSentId - mLastAppliedId < mChannel.maxUpdatesAhead;
  if (post) {
    if (mChannel.update) { // Previous update not yet picked up by the display, merge
      auto& update = *mChannel.update;
      update.keys.insert(update.keys.end(), mUpdate.keys.begin(), mUpdate.keys.end());
      if (mUpdate.lineWidth) {
        update.lineWidth = mUpdate.lineWidth;
      }
    } else {
      mChannel.update = std::move(mUpdate);
    }
    mChannel.update->id = ++mLastSentId;
    mChannel.guiSleeping = true;
    mUpdate = {};
  }
  lock.unlock();

  if (post) {
    mChannel.updatePosted();
  }
  if (apply) {
    applySettings();
  }
}

void GPUDisplayGUI::applySettings()
{
  const QSignalBlocker blocker(mUi->lineWidthSlider); // Not a user change
  mUi->lineWidthSlider->setValue(std::lround(mState.cfg.cfgL.lineWidth * 10.f));
  mUi->lineWidthValue->setNum(mState.cfg.cfgL.lineWidth);
}

void GPUDisplayGUI::bringToFront()
{
  setWindowState((windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
  show();
  raise();
  activateWindow();
}

void GPUDisplayGUI::keyPressEvent(QKeyEvent* event)
{
  const int32_t key = translateKey(event);
  if (key < 0) {
    QMainWindow::keyPressEvent(event);
    return;
  }
  mUpdate.keys.emplace_back(key);
}

int32_t GPUDisplayGUI::translateKey(const QKeyEvent* event)
{
  switch (event->key()) {
    case Qt::Key_Up:
      return GPUDisplayFrontend::KEY_UP;
    case Qt::Key_Down:
      return GPUDisplayFrontend::KEY_DOWN;
    case Qt::Key_Left:
      return GPUDisplayFrontend::KEY_LEFT;
    case Qt::Key_Right:
      return GPUDisplayFrontend::KEY_RIGHT;
    case Qt::Key_PageUp:
      return GPUDisplayFrontend::KEY_PAGEUP;
    case Qt::Key_PageDown:
      return GPUDisplayFrontend::KEY_PAGEDOWN;
    case Qt::Key_Home:
      return GPUDisplayFrontend::KEY_HOME;
    case Qt::Key_End:
      return GPUDisplayFrontend::KEY_END;
    case Qt::Key_Insert:
      return GPUDisplayFrontend::KEY_INSERT;
    case Qt::Key_Escape:
      return GPUDisplayFrontend::KEY_ESCAPE;
    case Qt::Key_Return:
    case Qt::Key_Enter:
      return GPUDisplayFrontend::KEY_ENTER;
    default:
      break;
  }
  if (event->key() >= Qt::Key_F1 && event->key() <= Qt::Key_F12) {
    static_assert(GPUDisplayFrontend::KEY_F12 - GPUDisplayFrontend::KEY_F1 == Qt::Key_F12 - Qt::Key_F1);
    return GPUDisplayFrontend::KEY_F1 + (event->key() - Qt::Key_F1);
  }
  const QString text = event->text();
  if (text.size() == 1 && text[0].unicode() >= 32 && text[0].unicode() < 127) { // Printable ASCII, with shift applied
    return text[0].unicode();
  }
  return -1;
}
