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
#include "PaletteHelpers.h"

namespace o2::framework
{

// Vivid accent colors — macOS system color palette / Pantone-adjacent
// RED: macOS Red (#FF3B30) / Pantone 485 C adjacent
const ImVec4 PaletteHelpers::RED = ImVec4(0xff / 255., 0x3b / 255., 0x30 / 255., 1);
// GREEN: macOS Green (#34C759) / Pantone 368 C adjacent
const ImVec4 PaletteHelpers::GREEN = ImVec4(0x34 / 255., 0xc7 / 255., 0x59 / 255., 1);
// BLUE: macOS Blue (#007AFF) / Pantone 2728 C adjacent
const ImVec4 PaletteHelpers::BLUE = ImVec4(0x00 / 255., 0x7a / 255., 0xff / 255., 1);
// YELLOW: macOS Yellow (#FFCC00) / Pantone 116 C adjacent
const ImVec4 PaletteHelpers::YELLOW = ImVec4(0xff / 255., 0xcc / 255., 0x00 / 255., 1);
// Muted/shaded variants — desaturated for secondary use
const ImVec4 PaletteHelpers::SHADED_RED = ImVec4(0xff / 255., 0x69 / 255., 0x61 / 255., 1);
const ImVec4 PaletteHelpers::SHADED_GREEN = ImVec4(0x86 / 255., 0xd9 / 255., 0x88 / 255., 1);
const ImVec4 PaletteHelpers::SHADED_BLUE = ImVec4(0x5a / 255., 0xc8 / 255., 0xfa / 255., 1);
const ImVec4 PaletteHelpers::SHADED_YELLOW = ImVec4(0xff / 255., 0xd6 / 255., 0x0a / 255., 1);
// Dark variants — for title bars and hovered states
// DARK_RED: Pantone 485 C (#DA291C)
const ImVec4 PaletteHelpers::DARK_RED = ImVec4(0xda / 255., 0x29 / 255., 0x1c / 255., 1);
// DARK_GREEN: (#1E8449)
const ImVec4 PaletteHelpers::DARK_GREEN = ImVec4(0x1e / 255., 0x84 / 255., 0x49 / 255., 1);
// DARK_YELLOW: macOS Orange (#FF9F0A) / Pantone 137 C adjacent
const ImVec4 PaletteHelpers::DARK_YELLOW = ImVec4(0xff / 255., 0x9f / 255., 0x0a / 255., 1);
// Neutrals — macOS dark mode system backgrounds
// WHITE: used as primary text / highlight color in dark UI
const ImVec4 PaletteHelpers::WHITE = ImVec4(0xf5 / 255., 0xf5 / 255., 0xf7 / 255., 1);
// BLACK: macOS dark background (#1C1C1E)
const ImVec4 PaletteHelpers::BLACK = ImVec4(0x1c / 255., 0x1c / 255., 0x1e / 255., 1);
// GRAY: macOS secondary background (#2C2C2E)
const ImVec4 PaletteHelpers::GRAY = ImVec4(0x2c / 255., 0x2c / 255., 0x2e / 255., 1);
// LIGHT_GRAY: macOS tertiary background (#3A3A3C)
const ImVec4 PaletteHelpers::LIGHT_GRAY = ImVec4(0x3a / 255., 0x3a / 255., 0x3c / 255., 1);

} // namespace o2::framework
