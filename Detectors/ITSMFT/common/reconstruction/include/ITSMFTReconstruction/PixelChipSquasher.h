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

/// \file   PixelChipSquasher.h
/// \brief  Merge persistent pixel information across ROFs due to ALPIDE singal time walking
/// \author matteo.concas@cern.ch

#ifndef O2_PIXEL_SQUASHER
#define O2_PIXEL_SQUASHER

#include <vector>

namespace o2
{
namespace itsmft
{
class Digit;
class ROFRecord;
class PixelChipSquasher
{
 public:
  PixelChipSquasher() = default;
  void process(std::vector<Digit>&, std::vector<ROFRecord>&);

 private:
};
} // namespace itsmft
} // namespace o2

#endif