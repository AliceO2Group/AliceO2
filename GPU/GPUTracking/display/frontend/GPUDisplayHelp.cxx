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

/// \file GPUDisplayHelp.cxx
/// \author David Rohr

#include "GPUDisplay.h"
#include "GPULogging.h"

using namespace o2::gpu;

void GPUDisplay::PrintHelp()
{
  mInfoHelpTimer.ResetStart();
  for (const auto& line : mFrontend->getHelpText()) {
    GPUInfo("%s", line.c_str());
  }
}

void GPUDisplay::PrintGLHelpText(float colorValue)
{
  const auto& lines = mFrontend->getHelpText();
  for (uint32_t i = 0; i < lines.size(); i++) {
    OpenGLPrint(lines[i].c_str(), 40.f, 35 + std::max(20, mDrawTextFontSize + 4) * (1 + i), colorValue, colorValue, colorValue, mInfoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - mInfoHelpTimer.GetCurrentElapsedTime()) : 1, false);
  }
}
