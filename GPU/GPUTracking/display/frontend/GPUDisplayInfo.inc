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

/// \file GPUDisplayInfo.inc
/// \author David Rohr

#ifndef GPUDISPLAYINFO_INC_H
#define GPUDISPLAYINFO_INC_H

#include "GPUDisplay.h"
#include "GPULogging.h"
#include "frontend/GPUDisplayConsole.h"

namespace o2::gpu
{
template <typename... Args>
void GPUDisplay::SetInfo(Args... args)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wformat-truncation"
  snprintf(mInfoText2, 1024, args...);
#pragma GCC diagnostic pop
  GPUInfo("%s", mInfoText2);
  mFrontend->mConsole->consoleInfo(std::get<0>(std::tuple<Args...>(args...)), mInfoText2);
  mGUIInfoText = mInfoText2;
  mGUIInfoTextCounter++;
  mInfoText2Timer.ResetStart();
}
} // namespace o2::gpu

#endif
