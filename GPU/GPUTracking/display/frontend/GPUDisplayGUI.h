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

#include "GPUDisplayGUIChannel.h"
#include <QMainWindow>
#include <QTimer>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

class QHBoxLayout;
class QKeyEvent;
class QLabel;
class QLineEdit;

namespace o2::gpu
{
class GPUDisplayGUI : public QMainWindow
{
  Q_OBJECT

 public:
  explicit GPUDisplayGUI(GPUDisplayGUIChannel& channel, QWidget* parent = nullptr);
  void bringToFront();

 protected:
  void keyPressEvent(QKeyEvent* event) override;
  void resizeEvent(QResizeEvent* event) override;
  bool eventFilter(QObject* watched, QEvent* event) override; // Ctrl + mouse wheel / Ctrl + 0 for all widgets of the window

 private:
  struct ElementWidget { // Result of the create* functions for a GUI element
    QWidget* widget = nullptr;
    std::function<void(const GPUDisplayGUIValue&)> set;                   // Set the element to a value from the display, signals are blocked
    std::function<void(const GPUDisplayGUIChannel::Options&)> setOptions; // Update range / choices of elements with dynamic options
  };
  struct GUIElement { // By element id
    std::function<void(const GPUDisplayGUIValue&)> set;
    std::function<void(const GPUDisplayGUIChannel::Options&)> setOptions;
    std::function<GPUDisplayGUIValue(const GPUDisplayGUIChannel::Settings&)> get; // Value of the element from mState.cfg
    std::function<GPUDisplayGUIChannel::Options()> getOptions;                    // Called only while the display waits for it
    std::optional<GPUDisplayGUIValue> applied;                                    // Value last set from the display, empty after a change by the user: the display might not accept it
  };

  void timerTick(); // Exchange updates / settings with the display
  void applySettings();
  void applyElement(GUIElement& element); // Set the element from mState.cfg if the value changed
  void updateInfoLabels();
  void fitStatusLabel(); // Height needed for the wrapped text, otherwise the size hint of the label makes the status bar too high
  bool darkPalette() const;
  void setDarkMode(bool dark);                                                    // Requests the color scheme from the platform theme
  void colorSchemeChanged();                                                      // Style sheets and colors depending on the color scheme
  QWidget* buildElements(const std::vector<GPUDisplayGUIChannel::Frame>& layout); // One frame per add_gui_binding call, with a labeled element per option
  ElementWidget createCheckbox(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createTextbox(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createSlider(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createRadioButtons(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createTextEdit(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createButton(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createDropdown(const GPUDisplayGUIChannel::Element& el);
  ElementWidget createMultiSlider(const GPUDisplayGUIChannel::Element& el);
  QHBoxLayout* createSliderRow(ElementWidget& retVal, QWidget* slider, QLineEdit* edit, Qt::Alignment editAlignment = {});
  void setValue(uint32_t id, GPUDisplayGUIValue value); // Element changed by the user, sent with the next update
  void setScale(float scale);                           // Font size, fixed element sizes, and check / radio indicators (not sent to the display)
  static int32_t translateKey(const QKeyEvent* event);  // Qt key → display key code, -1 if not forwarded

  QTimer mTimer;
  GPUDisplayGUIChannel& mChannel;
  GPUDisplayGUIChannel::Update mUpdate; // Changes collected since the last tick
  GPUDisplayGUIChannel::State mState;   // Last state received from the display
  uint64_t mLastSentId = 0;             // Id of the last update sent
  uint64_t mLastAppliedId = 0;          // Id of the last update for which the display state was received
  std::vector<GUIElement> mElements;
  uint64_t mOptionsRevision = 0;                         // optionsRevision for which the option lambdas were called
  std::vector<std::function<void(float)>> mScaleSetters; // Sizes of widgets and layouts that depend on the GUI scale
  QLabel* mStatusLabel = nullptr;                        // Status line and, for infoTextTime of the channel, info message from the display, at the bottom of the window
  QTimer mInfoTextTimer;
  uint64_t mInfoTextCounter = 0;
  bool mDarkMode = false;
  float mScale = 1.f;
  float mBaseFontSize = 0.f; // Point size of the font at scale 1
};
} // namespace o2::gpu

#endif // GPUDISPLAYGUI_H
