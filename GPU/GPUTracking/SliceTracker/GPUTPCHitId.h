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

/// \file GPUTPCHitId.h
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#ifndef GPUTPCHITID_H
#define GPUTPCHITID_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCHitId
{
 public:
  GPUhd() void Set(int32_t row, int32_t hit) { mId = (hit << 8) | row; }
  GPUhd() int32_t RowIndex() const { return mId & 0xff; }
  GPUhd() int32_t HitIndex() const { return mId >> 8; }

 private:
  int32_t mId;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCHITID_H
