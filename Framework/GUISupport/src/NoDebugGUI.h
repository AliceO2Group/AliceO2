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
#ifndef FRAMEWORK_NODEBUGGUI_H
#define FRAMEWORK_NODEBUGGUI_H

#include <functional>

namespace o2
{
namespace framework
{

// The DebugGUI has been moved to a separate package, this is a dummy header file
// included when the DebugGUI package is not found or disabled.
static inline void* initGUI(const char* name)
{
  return nullptr;
}

static inline bool pollGUI(void* context, std::function<void(void)> guiCallback)
{
  // returns whether quit is requested, we return 'no'
  return false;
}
static inline void disposeGUI()
{
}

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_NODEBUGGUI_H
