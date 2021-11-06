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

#ifndef ALICEO2_COMMON_PAIR_H
#define ALICEO2_COMMON_PAIR_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace dataformats
{
// A messageable pair
template <typename F, typename S>
struct Pair {
  Pair() = default;
  Pair(F f, S s) : first(f), second(s) {}
  F first{};
  S second{};
  ClassDefNV(Pair, 1);
};

} // namespace dataformats
} // namespace o2

#endif /* ALICEO2_COMMON_PAIR_H */
