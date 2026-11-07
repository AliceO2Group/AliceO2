// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayGUIChannel.h
/// \author David Rohr

#ifndef GPUDISPLAYGUICHANNEL_H
#define GPUDISPLAYGUICHANNEL_H

#define GPUSETTINGS_DISPLAYCONTROL
#include "GPUSettings.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace o2::gpu
{
using GPUDisplayGUIValue = std::variant<int64_t, double, std::string>; // Value of a Qt GUI element

struct GPUDisplayGUIUpdate { // GUI → display
  uint64_t id = 0;
  std::vector<int32_t> keys;                                   // Keys pressed in the GUI and not handled by it, display key codes
  std::vector<std::pair<uint32_t, GPUDisplayGUIValue>> values; // Elements changed in the GUI: element id, new value
  std::optional<float> guiScale;                               // GUI scale changed with Ctrl + mouse wheel / Ctrl + 0
  std::optional<int8_t> guiDarkMode;                           // System color scheme, reported when the GUI starts if not yet set
  bool empty() const { return keys.empty() && values.empty() && !guiScale && !guiDarkMode; }
  void setValue(uint32_t element, GPUDisplayGUIValue value) // Replaces an earlier value of the same element
  {
    auto it = std::find_if(values.begin(), values.end(), [element](const auto& v) { return v.first == element; });
    if (it != values.end()) {
      it->second = std::move(value);
    } else {
      values.emplace_back(element, std::move(value));
    }
  }
  void merge(const GPUDisplayGUIUpdate& other) // Newer changes on top of this update, id not changed
  {
    keys.insert(keys.end(), other.keys.begin(), other.keys.end());
    for (const auto& [element, value] : other.values) {
      setValue(element, value);
    }
    guiScale = other.guiScale ? other.guiScale : guiScale;
    guiDarkMode = other.guiDarkMode ? other.guiDarkMode : guiDarkMode;
  }
};

// Communication between the Qt GUI thread (GPUDisplayGUI) and the display thread (GPUDisplayFrontend::HandleExternalGUISignals), Qt-free.
// The GUI sends at most one update per timer tick, the display processes it once per frame and publishes the complete settings with the id of the last processed update.
// If the GUI went to sleep with an update ready and the display picks it up, the GUI waits for the reply when it wakes up, so it continues with the settings including its update.
struct GPUDisplayGUIChannel {
  struct Settings { // Copy of the config structs of GPUDisplay, with the same names, the GUI reads the values of its elements from it
    GPUSettingsDisplayHeavy mCfgH;
    GPUSettingsDisplayLight mCfgL;
    GPUSettingsDisplayRenderer mCfgR;
    GPUDisplayControl mCfgC;
  };
  enum class ElementType : uint8_t { CHECKBOX,
                                     TEXTBOX,
                                     SLIDER,
                                     RADIO_BUTTONS,
                                     DROPDOWN,
                                     BUTTON,
                                     TEXTEDIT };

  struct Options { // Dynamic range (slider) or choices (dropdown) of an element, returned by its option lambda
    double min = 0, max = 0;
    std::vector<std::pair<std::string, GPUDisplayGUIValue>> choices;
  };
  struct Element {                                                   // GUI element for one option, from GUI_LAMBDA in GPUDisplayFrontendConfigQtGUI.inc
    std::string name;                                                // Label
    std::string description;                                         // Tooltip, from the RTTI description of the option
    uint32_t rttiIndex;                                              // Option in the display config RTTI, invalid for options without RTTI (GPUDisplayControl)
    ElementType type;                                                // Widget type
    uint32_t id;                                                     // Index of the element in Update::values
    bool isInteger;                                                  // Integer / bool option, otherwise floating point
    bool isString = false;                                           // String option (text field sends the text)
    bool horizontal = false;                                         // In a row with the neighbouring horizontal elements, wrapping at the window border, otherwise an own row
    uint32_t lines = 0;                                              // Text edit: initial number of lines
    bool hasRange = false;                                           // Text field: value limited to min / max
    double min = 0, max = 0;                                         // Slider range / text field range (if hasRange), from the min / max of the option
    std::vector<std::pair<std::string, GPUDisplayGUIValue>> choices; // Radio buttons / dropdown / slider with only these values: name -> value
    std::function<Options()> dynamicOptions;                         // Called by the GUI only while the display waits for it (see optionsRevision), reads display data
    std::function<GPUDisplayGUIValue(const Settings&)> get;          // Value of the option from a copy of the settings, callable from the GUI thread
  };
  struct Frame { // One add_gui_binding call
    std::string description;
    std::vector<Element> elements;
  };
  using Update = GPUDisplayGUIUpdate;
  struct State { // display → GUI
    Settings cfg;
    std::string infoStatus;       // Status line for the bottom of the GUI, empty if not sent
    std::string infoText;         // Latest info message, empty if not sent
    uint64_t infoTextCounter = 0; // Incremented with every info message
    uint64_t optionsRevision = 0; // Incremented when the GUI must call the dynamic option lambdas again (new event, GUI started)
    uint64_t appliedUpdate = 0;   // Id of the last GUI update included in cfg
    uint64_t counter = 0;         // Incremented with every published state
  };

  std::mutex mutex;                            // Protects all members below
  std::condition_variable stateUpdated;        // Notified when a new state is published or the GUI shall stop
  std::condition_variable optionsRead;         // Notified by the GUI when it called the dynamic option lambdas
  uint64_t optionsReadRevision = 0;            // optionsRevision for which the GUI called the dynamic option lambdas
  bool displayWaitingForOptions = false;       // Display waits until the GUI called the dynamic option lambdas, only then the GUI may call them
  std::optional<Update> update;                // Pending update, not yet picked up by the display
  State state;                                 // Latest state published by the display
  bool guiSleeping = false;                    // GUI waits for its next timer tick with an update ready
  bool displayProcessingCurrentUpdate = false; // Display processes the update the GUI went to sleep with, GUI waits for the reply
  bool stop = false;                           // GUI shall close
  std::chrono::milliseconds replyTimeout;      // Maximum time the GUI waits for a reply
  uint32_t maxUpdatesAhead = 0;                // GUI waits for the reply if it is that many updates ahead of the last one it applied
  std::string windowName;                      // Name of the display window, set when starting the GUI
  std::vector<Frame> layout;                   // GUI elements, set when starting the GUI
  std::function<void()> updatePosted;          // Set by the display, called by the GUI (without holding mutex) after posting an update, wakes up the display
};
} // namespace o2::gpu

#endif // GPUDISPLAYGUICHANNEL_H
