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
#ifndef O2_FRAMEWORK_PALETTEHELPERS_H_
#define O2_FRAMEWORK_PALETTEHELPERS_H_

#include "DebugGUI/imgui.h"

namespace o2::framework
{

/// An helper class for colors and palettes
struct PaletteHelpers {
  static const ImVec4 RED;
  static const ImVec4 GREEN;
  static const ImVec4 BLUE;
  static const ImVec4 YELLOW;
  static const ImVec4 SHADED_RED;
  static const ImVec4 SHADED_GREEN;
  static const ImVec4 SHADED_BLUE;
  static const ImVec4 SHADED_YELLOW;
  static const ImVec4 DARK_RED;
  static const ImVec4 DARK_GREEN;
  static const ImVec4 DARK_YELLOW;
  static const ImVec4 WHITE;
  static const ImVec4 BLACK;
  static const ImVec4 GRAY;
  static const ImVec4 LIGHT_GRAY;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PALETTEHELPERS_H_
