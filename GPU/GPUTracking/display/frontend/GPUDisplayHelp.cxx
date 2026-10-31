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

#include <string>

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
  const int32_t maxChars = mBackend->mFontCharWidth > 0.f ? (mBackend->mScreenWidth / mFrontend->getDisplayScale() - 80.f) / mBackend->mFontCharWidth : 0; // Characters fitting between the margins, 0 = unknown (no Freetype font), no wrapping
  uint32_t row = 0;
  auto print = [&](const char* s, uint32_t nCharacters, uint32_t indent) { GLPrintText(s, 40.f + indent * mBackend->mFontCharWidth, 35 + std::max(20, mDrawTextFontSize + 4) * (1 + row++), colorValue, colorValue, colorValue, mInfoHelpTimer.GetCurrentElapsedTime() >= 5 ? (6 - mInfoHelpTimer.GetCurrentElapsedTime()) : 1, nCharacters, false); };
  for (const auto& line : lines) {
    if (maxChars <= 0 || line.size() <= (size_t)maxChars) {
      print(line.c_str(), 0, 0);
      continue;
    }
    const size_t textStart = line.find(": ");
    const size_t indent = textStart != std::string::npos && textStart + 2 < (size_t)maxChars / 2 ? textStart + 2 : 0; // Continuation lines aligned with the text after the keys, if that leaves enough space
    size_t pos = 0;                                                                                                   // Start of the part not yet printed
    for (bool first = true;; first = false) {                                                                         // Wrap at the last space that fits, or hard if there is none (also if it is only in the keys)
      const size_t lineIndent = first ? 0 : indent, available = maxChars - lineIndent;
      if (line.size() - pos <= available) {
        print(line.c_str() + pos, line.size() - pos, lineIndent);
        break;
      }
      size_t cut = line.rfind(' ', pos + available);
      if (cut == std::string::npos || cut <= pos + (first ? indent : 0)) {
        cut = pos + available;
      }
      print(line.c_str() + pos, cut - pos, lineIndent);
      pos = line[cut] == ' ' ? cut + 1 : cut;
    }
  }
}
