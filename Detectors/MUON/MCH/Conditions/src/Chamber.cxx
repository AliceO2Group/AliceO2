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

#include "MCHConditions/Chamber.h"
#include <fmt/core.h>

namespace o2::mch::dcs
{
std::optional<Chamber> chamber(int chamberId)
{
  switch (chamberId) {
    case 0:
      return Chamber::Ch00;
    case 1:
      return Chamber::Ch01;
    case 2:
      return Chamber::Ch02;
    case 3:
      return Chamber::Ch03;
    case 4:
      return Chamber::Ch04;
    case 5:
      return Chamber::Ch05;
    case 6:
      return Chamber::Ch06;
    case 7:
      return Chamber::Ch07;
    case 8:
      return Chamber::Ch08;
    case 9:
      return Chamber::Ch09;
  }
  return std::nullopt;
}

bool isQuadrant(Chamber chamber)
{
  return (
    chamber == Chamber::Ch00 ||
    chamber == Chamber::Ch01 ||
    chamber == Chamber::Ch02 ||
    chamber == Chamber::Ch03);
}

bool isSlat(Chamber chamber)
{
  return !isQuadrant(chamber);
}

bool isStation1(Chamber chamber)
{
  return chamber == Chamber::Ch00 || chamber == Chamber::Ch01;
}

bool isStation2(Chamber chamber)
{
  return chamber == Chamber::Ch02 || chamber == Chamber::Ch03;
}

int toInt(Chamber chamberId)
{
  switch (chamberId) {
    case Chamber::Ch00:
      return 0;
    case Chamber::Ch01:
      return 1;
    case Chamber::Ch02:
      return 2;
    case Chamber::Ch03:
      return 3;
    case Chamber::Ch04:
      return 4;
    case Chamber::Ch05:
      return 5;
    case Chamber::Ch06:
      return 6;
    case Chamber::Ch07:
      return 7;
    case Chamber::Ch08:
      return 8;
    case Chamber::Ch09:
      return 9;
  }
  return -1; // to make GCC happy
}

std::string name(Chamber chamber)
{
  return fmt::format("Ch{:02d}", toInt(chamber));
}

Chamber aliasToChamber(std::string_view alias)
{
  std::string ch{"Chamber"};
  auto pos = alias.find(ch);
  std::string s{alias.substr(pos + ch.size(), 2)};
  int chamberId = std::stoi(s);
  return chamber(chamberId).value();
}

} // namespace o2::mch::dcs
