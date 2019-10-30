// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>

#include <nonstd/span.h>

#include <sstream>
#include <vector>

namespace gpucf
{

class DigitDrawer
{

 public:
  DigitDrawer(
    nonstd::span<const Digit>,
    nonstd::span<unsigned char>,
    nonstd::span<unsigned char>);

  DigitDrawer(
    nonstd::span<const Digit>,
    nonstd::span<const Digit>,
    nonstd::span<const Digit>);

  std::string drawArea(const Digit&, int r);

 private:
  Map<float> chargeMap;
  Map<unsigned char> peakGTMap;
  Map<unsigned char> peakMap;

  std::string toFixed(float);

  void printAt(std::stringstream&, const Position&);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
