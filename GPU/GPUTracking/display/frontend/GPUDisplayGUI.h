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

/// \file GPUDisplayGUI.h
/// \author David Rohr

#ifndef GPUDISPLAYGUI_H
#define GPUDISPLAYGUI_H

#define QT_BEGIN_NAMESPACE \
  namespace o2::gpu::qtgui \
  {
#define QT_END_NAMESPACE }

#include "GPUDisplayGUIChannel.h"
#include <QMainWindow>
#include <QTimer>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

class QKeyEvent;

QT_BEGIN_NAMESPACE
namespace Ui
{
class GPUDisplayGUI;
}
QT_END_NAMESPACE

namespace o2::gpu
{
class GPUDisplayGUI : public QMainWindow
{
  Q_OBJECT

 public:
  explicit GPUDisplayGUI(GPUDisplayGUIChannel& channel, QWidget* parent = nullptr);
  ~GPUDisplayGUI() override;
  void bringToFront();

 protected:
  void keyPressEvent(QKeyEvent* event) override;
  bool eventFilter(QObject* watched, QEvent* event) override; // Ctrl + mouse wheel / Ctrl + 0 for all widgets of the window

 private:
  void timerTick(); // Exchange updates / settings with the display
  void applySettings();
  void buildElements(const std::vector<GPUDisplayGUIChannel::Frame>& layout); // One frame per add_gui_binding call, with a labeled element per option
  void setValue(uint32_t id, GPUDisplayGUIValue value);                       // Element changed by the user, sent with the next update
  void setScale(float scale);                                                 // Font size, fixed element sizes, and check / radio indicators (not sent to the display)
  static int32_t translateKey(const QKeyEvent* event);                        // Qt key → display key code, -1 if not forwarded

  std::unique_ptr<Ui::GPUDisplayGUI> mUi;
  QTimer mTimer;
  GPUDisplayGUIChannel& mChannel;
  GPUDisplayGUIChannel::Update mUpdate;                                        // Changes collected since the last tick
  GPUDisplayGUIChannel::State mState;                                          // Last state received from the display
  uint64_t mLastSentId = 0;                                                    // Id of the last update sent
  uint64_t mLastAppliedId = 0;                                                 // Id of the last update for which the display state was received
  std::vector<std::function<void(const GPUDisplayGUIValue&)>> mElementSetters; // By element id, set the element to a value from the display
  struct ScaledWidth {
    QWidget* widget;
    int width;  // Unscaled pixels
    bool fixed; // Fixed width, otherwise minimum width
  };
  std::vector<ScaledWidth> mScaledWidths; // Widget widths that scale with the GUI
  std::vector<QWidget*> mSliderFrames;    // Frames with floating point sliders, preferred width 1/3 of the screen width times the scale
  QLayout* mFramesLayout = nullptr;       // FlowLayout of the frames
  float mScale = 1.f;
  float mBaseFontSize = 0.f; // Point size of the font at scale 1
};
} // namespace o2::gpu

#endif // GPUDISPLAYGUI_H
