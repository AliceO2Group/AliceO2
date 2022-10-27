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
#ifndef O2_FRAMEWORK_NODEBUGGUI_H_
#define O2_FRAMEWORK_NODEBUGGUI_H_

#include <functional>

namespace o2::framework
{

// The DebugGUI has been moved to a separate package, this is a dummy header file
// included when the DebugGUI package is not found or disabled.
static inline void* initGUI(const char* name)
{
  return nullptr;
}

static inline void disposeGUI()
{
}

static inline void getFrameJSON(void* data, std::ostream& json_data) override
{
}

static inline void getFrameRaw(void* data, void** raw_data, int* size) override
{
}

static inline bool pollGUIPreRender(void* context, float delta) override
{
  return true;
}

static inline void* pollGUIRender(std::function<void(void)> guiCallback) override
{
  return nullptr;
}

static inline void pollGUIPostRender(void* context, void* draw_data) override
{
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_NODEBUGGUI_H_
