// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ZDC_MODULECONFIG_H
#define ALICEO2_ZDC_MODULECONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

namespace o2
{
namespace zdc
{

struct TriggerChannelConfig {
  int8_t id = -1;
  int8_t first = 0;
  int8_t last = 0;
  uint8_t shift = 0;
  int16_t threshold = 0;
  void print() const;
  ClassDefNV(TriggerChannelConfig, 1);
};

struct Module {
  static constexpr int MaxChannels = 4, MaxTriggChannels = 2;
  int id = -1; // not active
  std::array<int8_t, MaxChannels> channelID = {IdDummy, IdDummy, IdDummy, IdDummy};
  std::array<int16_t, MaxChannels> linkID = {-1, -1, -1, -1};
  std::array<bool, MaxChannels> readChannel = {false};
  std::array<bool, MaxChannels> trigChannel = {false};
  std::array<TriggerChannelConfig, MaxChannels> trigChannelConf;

  void setChannel(int slot, int8_t chID, int16_t lID, bool read, bool trig = false, int tF = 0, int tL = 0, int tS = 0, int tT = 0);
  void print() const;
  void check() const;
  ClassDefNV(Module, 1);
};

struct ModuleConfig {
  static constexpr int MaxNModules = 8;
  std::array<Module, MaxNModules> modules;

  void print() const;
  void check() const;
  ClassDefNV(ModuleConfig, 1);
};

} // namespace zdc
} // namespace o2

#endif
