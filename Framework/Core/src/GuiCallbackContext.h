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

namespace o2::framework
{

struct GuiCallbackContext;
class WSDPLHandler;

struct GuiRenderer {
  uv_timer_t drawTimer;
  WSDPLHandler* handler;
  GuiCallbackContext* gui;
};

struct GuiCallbackContext {
  uint64_t frameLast;
  float* frameLatency;
  float* frameCost;
  void* lastFrame;
  DebugGUI* plugin;
  void* window;
  bool* guiQuitRequested;
  std::function<void(void)> callback;
  std::set<GuiRenderer*> renderers;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_GUICALLBACKCONTEXT_H_
