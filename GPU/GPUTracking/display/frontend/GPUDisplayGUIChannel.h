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

#include "GPUSettings.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace o2::gpu
{
// Communication between the Qt GUI thread (GPUDisplayGUI) and the display thread (GPUDisplayFrontend::HandleExternalGUISignals), Qt-free.
// The GUI sends at most one update per timer tick, the display processes it once per frame and publishes the complete settings with the id of the last processed update.
// If the GUI went to sleep with an update ready and the display picks it up, the GUI waits for the reply when it wakes up, so it continues with the settings including its update.
struct GPUDisplayGUIChannel {
  struct Settings {
    GPUSettingsDisplayHeavy cfgH;
    GPUSettingsDisplayLight cfgL;
    GPUSettingsDisplayRenderer cfgR;
  };
  struct Update { // GUI → display
    uint64_t id = 0;
    std::vector<int32_t> keys;      // Keys pressed in the GUI and not handled by it, display key codes
    std::optional<float> lineWidth; // For testing, to be replaced by generic settings updates
    bool empty() const { return keys.empty() && !lineWidth; }
  };
  struct State { // display → GUI
    Settings cfg;
    uint64_t appliedUpdate = 0; // Id of the last GUI update included in cfg
    uint64_t counter = 0;       // Incremented with every published state
  };

  std::mutex mutex;                            // Protects all members below
  std::condition_variable stateUpdated;        // Notified when a new state is published or the GUI shall stop
  std::optional<Update> update;                // Pending update, not yet picked up by the display
  State state;                                 // Latest state published by the display
  bool guiSleeping = false;                    // GUI waits for its next timer tick with an update ready
  bool displayProcessingCurrentUpdate = false; // Display processes the update the GUI went to sleep with, GUI waits for the reply
  bool stop = false;                           // GUI shall close
  std::chrono::milliseconds replyTimeout;      // Maximum time the GUI waits for a reply
  uint32_t maxUpdatesAhead = 0;                // GUI waits for the reply if it is that many updates ahead of the last one it applied
  std::string windowName;                      // Name of the display window, set when starting the GUI
  std::function<void()> updatePosted;          // Set by the display, called by the GUI (without holding mutex) after posting an update, wakes up the display
};
} // namespace o2::gpu

#endif // GPUDISPLAYGUICHANNEL_H
