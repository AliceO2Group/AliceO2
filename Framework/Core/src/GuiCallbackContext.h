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

#include <functional>

namespace o2::framework
{

struct GuiRenderer {
  uint64_t latency;
  uint64_t frameLast;
  std::function<void(void*,int)> drawCallback;
};

struct GuiCallbackContext {
  uint64_t frameLast;
  float* frameLatency;
  float* frameCost;
  DebugGUI* plugin;
  void* window;
  bool* guiQuitRequested;
  std::function<void(void)> callback;
  std::map<std::string, GuiRenderer*> renderers;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_GUICALLBACKCONTEXT_H_
