// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_PALETTE_HELPER_H
#define FRAMEWORK_PALETTE_HELPER_H

#include "DebugGUI/imgui.h"

namespace o2
{
namespace framework
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

} // namespace framework
} // namespace o2

#endif
