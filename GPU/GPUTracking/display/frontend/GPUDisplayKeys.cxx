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

/// \file GPUDisplayKeys.cxx
/// \author David Rohr

#include "GPUDisplay.h"
#include "frontend/GPUDisplayInfo.inc"

using namespace o2::gpu;

const char* HelpText[] = {
  "[MOUSE 1]                     Look around",
  "[MOUSE 2]                     Strafe camera",
  "[MOUSE 1+2]                   Zoom / Rotate",
  "[SHIFT]                       Slow Zoom / Move / Rotate",
  // FREE: [SPACE] [Q]
  // Test setting: ^
};

void GPUDisplay::PrintHelp()
{
  mInfoHelpTimer.ResetStart();
  for (uint32_t i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    GPUInfo("%s", HelpText[i]);
  }
}

void GPUDisplay::HandleKey(uint8_t key, bool doNotHandleConfigChanges)
{
  const GPUSettingsDisplayHeavy oldCfgH = mCfgH;
  const GPUSettingsDisplayLight oldCfgL = mCfgL;
  const GPUSettingsDisplayRenderer oldCfgR = mCfgR;
  if (key == mFrontend->KEY_ALT) {
    mFrontend->mKeys[mFrontend->KEY_CTRL] = false; // Release CTRL with alt, to avoid orienting along y automatically!
  } else if (mFrontend->processKeyLambda(key)) {
  }
  /*
  else if (key == '^')
  {
    mTestSetting++;
    SetInfo("Debug test variable set to %d", mTestSetting);
  }
  */

  if (!doNotHandleConfigChanges) {
    checkAndApplyConfigChanges(oldCfgH, oldCfgL, oldCfgR);
  }
}

void GPUDisplay::PrintGLHelpText(float colorValue)
{
  for (uint32_t i = 0; i < sizeof(HelpText) / sizeof(HelpText[0]); i++) {
    OpenGLPrint(HelpText[i], 40.f, 35 + std::max(20, mDrawTextFontSize + 4) * (1 + i), colorValue, colorValue, colorValue, mInfoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - mInfoHelpTimer.GetCurrentElapsedTime()) : 1, false);
  }
}
