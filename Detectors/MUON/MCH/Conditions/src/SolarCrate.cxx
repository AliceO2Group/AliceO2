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

///
/// GENERATED CODE ! DO NOT EDIT !
///

#include "MCHConditions/SolarCrate.h"
#include <map>
#include <string>
#include <fmt/core.h>

namespace o2::mch::dcs
{
int aliasToSolarCrate(std::string_view alias)
{
  static const std::map<std::string, int> a2c = {
    {"MchHvLvLeft/Chamber04Left/SolCh04LCr01", 18},
    {"MchHvLvLeft/Chamber04Left/SolCh04LCr02", 10},
    {"MchHvLvLeft/Chamber04Left/SolCh04LCr03", 53},
    {"MchHvLvLeft/Chamber04Left/SolCh04LCr04", 55},
    {"MchHvLvLeft/Chamber04Left/SolCh04LCr05", 42},
    {"MchHvLvLeft/Chamber05Left/SolCh05LCr01", 39},
    {"MchHvLvLeft/Chamber05Left/SolCh05LCr02", 1},
    {"MchHvLvLeft/Chamber05Left/SolCh05LCr03", 7},
    {"MchHvLvLeft/Chamber05Left/SolCh05LCr04", 4},
    {"MchHvLvLeft/Chamber05Left/SolCh05LCr05", 3},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr01", 90},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr02", 115},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr03", 98},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr04", 114},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr05", 41},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr06", 43},
    {"MchHvLvLeft/Chamber06Left/SolCh06LCr07", 106},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr01", 91},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr02", 92},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr03", 97},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr04", 108},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr05", 44},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr06", 38},
    {"MchHvLvLeft/Chamber07Left/SolCh07LCr07", 107},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr01", 85},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr02", 93},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr03", 94},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr04", 87},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr05", 88},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr06", 79},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr07", 110},
    {"MchHvLvLeft/Chamber08Left/SolCh08LCr08", 109},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr01", 80},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr02", 89},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr03", 96},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr04", 82},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr05", 95},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr06", 113},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr07", 111},
    {"MchHvLvLeft/Chamber09Left/SolCh09LCr08", 112},
    {"MchHvLvRight/Chamber04Right/SolCh04RCr01", 9},
    {"MchHvLvRight/Chamber04Right/SolCh04RCr02", 23},
    {"MchHvLvRight/Chamber04Right/SolCh04RCr03", 57},
    {"MchHvLvRight/Chamber04Right/SolCh04RCr04", 50},
    {"MchHvLvRight/Chamber04Right/SolCh04RCr05", 46},
    {"MchHvLvRight/Chamber05Right/SolCh05RCr01", 56},
    {"MchHvLvRight/Chamber05Right/SolCh05RCr02", 45},
    {"MchHvLvRight/Chamber05Right/SolCh05RCr03", 27},
    {"MchHvLvRight/Chamber05Right/SolCh05RCr04", 54},
    {"MchHvLvRight/Chamber05Right/SolCh05RCr05", 51},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr01", 105},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr02", 100},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr03", 102},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr04", 78},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr05", 66},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr06", 64},
    {"MchHvLvRight/Chamber06Right/SolCh06RCr07", 73},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr01", 103},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr02", 101},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr03", 104},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr04", 75},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr05", 72},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr06", 74},
    {"MchHvLvRight/Chamber07Right/SolCh07RCr07", 62},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr01", 86},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr02", 84},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr03", 70},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr04", 71},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr05", 76},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr06", 77},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr07", 60},
    {"MchHvLvRight/Chamber08Right/SolCh08RCr08", 61},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr01", 81},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr02", 83},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr03", 69},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr04", 63},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr05", 67},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr06", 68},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr07", 65},
    {"MchHvLvRight/Chamber09Right/SolCh09RCr08", 59},
  };
  int i = alias.find('.');
  std::string salias(alias.substr(0, i));
  auto p = a2c.find(salias);
  if (p != a2c.end()) {
    return p->second;
  }
  throw std::invalid_argument(fmt::format("Cannot extract solar create from alias={}", alias));
}
} // namespace o2::mch::dcs
