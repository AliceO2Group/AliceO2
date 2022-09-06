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

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FDD reconstruction
#ifndef ALICEO2_FDD_RECONSTRUCTOR_H
#define ALICEO2_FDD_RECONSTRUCTOR_H

#include <vector>
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/RecPoint.h"
namespace o2
{
namespace fdd
{
class Reconstructor
{
 public:
  Reconstructor() = default;
  ~Reconstructor() = default;
  void process(o2::fdd::Digit const& digitBC,
               gsl::span<const o2::fdd::ChannelData> inChData,
               std::vector<o2::fdd::RecPoint>& RecPoints,
               std::vector<o2::fdd::ChannelDataFloat>& outChData);

  void finish();

 private:
  ClassDefNV(Reconstructor, 3);
};
} // namespace fdd
} // namespace o2
#endif
