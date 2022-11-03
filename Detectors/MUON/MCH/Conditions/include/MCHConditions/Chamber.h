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

#ifndef O2_MCH_CONDITIONS_CHAMBER_H
#define O2_MCH_CONDITIONS_CHAMBER_H

#include <optional>
#include <string>

namespace o2::mch::dcs
{

/** The possible chamber numbers */
enum class Chamber {
  Ch00,
  Ch01,
  Ch02,
  Ch03,
  Ch04,
  Ch05,
  Ch06,
  Ch07,
  Ch08,
  Ch09
};

/** name of Chamber */
std::string name(Chamber chamber);

/** convert chamber to a plain integer */
int toInt(Chamber chamber);

/** convert (if possible) a chamber to an integer */
std::optional<Chamber> chamber(int chamberId);

/** extract the chamber information from the alias.
 * alias must be valid otherwise the method throws an exception. */
Chamber aliasToChamber(std::string_view alias);

bool isSlat(Chamber chamber);

bool isQuadrant(Chamber chamber);

bool isStation1(Chamber chamber);
bool isStation2(Chamber chamber);

} // namespace o2::mch::dcs
#endif
