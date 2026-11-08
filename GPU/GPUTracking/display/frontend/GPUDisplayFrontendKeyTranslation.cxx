// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendKeyTranslation.cxx
/// \author David Rohr

#include "GPUDisplayFrontend.h"
#include <utility>

using namespace o2::gpu;

static constexpr uint64_t encodeSequence(const char* s) // As encoded by the console: first character in the lowest byte
{
  uint64_t retVal = 0;
  for (uint32_t i = 0; i < 8 && s[i]; i++) {
    retVal |= (uint64_t)(uint8_t)s[i] << (8 * i);
  }
  return retVal;
}

int32_t GPUDisplayFrontend::translateExternalKey(uint32_t translationMode, uint64_t extendedKey)
{
  if (translationMode == 1) {
    // Console escape sequences, characters after the ESC: VT100 / xterm, cursor keys also in application mode (O instead of [), F1 to F5 also of rxvt ([11~) and the Linux console ([[A), with SHIFT of xterm (;2) and rxvt ([a, $)
    static constexpr std::pair<const char*, int32_t> sequences[] = {
      {"[A", KEY_UP},
      {"[B", KEY_DOWN},
      {"[C", KEY_RIGHT},
      {"[D", KEY_LEFT},
      {"OA", KEY_UP},
      {"OB", KEY_DOWN},
      {"OC", KEY_RIGHT},
      {"OD", KEY_LEFT},
      {"[H", KEY_HOME},
      {"[F", KEY_END},
      {"OH", KEY_HOME},
      {"OF", KEY_END},
      {"[1~", KEY_HOME},
      {"[4~", KEY_END},
      {"[7~", KEY_HOME},
      {"[8~", KEY_END},
      {"[2~", KEY_INSERT},
      {"[5~", KEY_PAGEUP},
      {"[6~", KEY_PAGEDOWN},
      {"OP", KEY_F1},
      {"OQ", KEY_F2},
      {"OR", KEY_F3},
      {"OS", KEY_F4},
      {"[11~", KEY_F1},
      {"[12~", KEY_F2},
      {"[13~", KEY_F3},
      {"[14~", KEY_F4},
      {"[[A", KEY_F1},
      {"[[B", KEY_F2},
      {"[[C", KEY_F3},
      {"[[D", KEY_F4},
      {"[[E", KEY_F5},
      {"[15~", KEY_F5},
      {"[17~", KEY_F6},
      {"[18~", KEY_F7},
      {"[19~", KEY_F8},
      {"[20~", KEY_F9},
      {"[21~", KEY_F10},
      {"[23~", KEY_F11},
      {"[24~", KEY_F12},
      {"[1;2A", KEY_UP + SPECIAL_KEYS_SHIFTED},
      {"[1;2B", KEY_DOWN + SPECIAL_KEYS_SHIFTED},
      {"[1;2C", KEY_RIGHT + SPECIAL_KEYS_SHIFTED},
      {"[1;2D", KEY_LEFT + SPECIAL_KEYS_SHIFTED},
      {"[1;2H", KEY_HOME + SPECIAL_KEYS_SHIFTED},
      {"[1;2F", KEY_END + SPECIAL_KEYS_SHIFTED},
      {"[2;2~", KEY_INSERT + SPECIAL_KEYS_SHIFTED},
      {"[5;2~", KEY_PAGEUP + SPECIAL_KEYS_SHIFTED},
      {"[6;2~", KEY_PAGEDOWN + SPECIAL_KEYS_SHIFTED},
      {"[a", KEY_UP + SPECIAL_KEYS_SHIFTED},
      {"[b", KEY_DOWN + SPECIAL_KEYS_SHIFTED},
      {"[c", KEY_RIGHT + SPECIAL_KEYS_SHIFTED},
      {"[d", KEY_LEFT + SPECIAL_KEYS_SHIFTED},
      {"[7$", KEY_HOME + SPECIAL_KEYS_SHIFTED},
      {"[8$", KEY_END + SPECIAL_KEYS_SHIFTED},
      {"[2$", KEY_INSERT + SPECIAL_KEYS_SHIFTED},
      {"[5$", KEY_PAGEUP + SPECIAL_KEYS_SHIFTED},
      {"[6$", KEY_PAGEDOWN + SPECIAL_KEYS_SHIFTED},
      {"[1;2P", KEY_F1 + SPECIAL_KEYS_SHIFTED},
      {"[1;2Q", KEY_F2 + SPECIAL_KEYS_SHIFTED},
      {"[1;2R", KEY_F3 + SPECIAL_KEYS_SHIFTED},
      {"[1;2S", KEY_F4 + SPECIAL_KEYS_SHIFTED},
      {"[15;2~", KEY_F5 + SPECIAL_KEYS_SHIFTED},
      {"[17;2~", KEY_F6 + SPECIAL_KEYS_SHIFTED},
      {"[18;2~", KEY_F7 + SPECIAL_KEYS_SHIFTED},
      {"[19;2~", KEY_F8 + SPECIAL_KEYS_SHIFTED},
      {"[20;2~", KEY_F9 + SPECIAL_KEYS_SHIFTED},
      {"[21;2~", KEY_F10 + SPECIAL_KEYS_SHIFTED},
      {"[23;2~", KEY_F11 + SPECIAL_KEYS_SHIFTED},
      {"[24;2~", KEY_F12 + SPECIAL_KEYS_SHIFTED},
    };
    for (const auto& [sequence, key] : sequences) {
      if (encodeSequence(sequence) == extendedKey) {
        return key;
      }
    }
  }
  return -1;
}
