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
/// Helper struct which holds all the lists the Driver needs to know about.
#ifndef O2_FRAMEWORK_GUICALLBACKCONTEXT_H_
#define O2_FRAMEWORK_GUICALLBACKCONTEXT_H_

#include "Framework/DebugGUI.h"
#include "Framework/DeviceState.h"

#include <functional>
#include <set>
#include <uv.h>

namespace o2::framework
{

struct GuiCallbackContext;
class WSDPLHandler;

struct GuiRenderer {
  uv_timer_t drawTimer;
  WSDPLHandler* handler = nullptr;
  GuiCallbackContext* gui = nullptr;
  bool guiConnected = false;
  /// Set to true if the renderer needs to update the textures
  /// in the next frame.
  bool updateTextures = true;
};

struct GuiCallbackContext {
  uint64_t frameLast;
  float* frameLatency = nullptr;
  float* frameCost = nullptr;
  void* lastFrame = nullptr;
  DebugGUI* plugin = nullptr;
  void* window = nullptr;
  bool* guiQuitRequested = nullptr;
  bool* allChildrenGone = nullptr;
  std::function<void(void)> callback;
  std::set<GuiRenderer*> renderers;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_GUICALLBACKCONTEXT_H_
