// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  GPUhd() void Set(int row, int hit) { mId = (hit << 8) | row; }
  GPUhd() int RowIndex() const { return mId & 0xff; }
  GPUhd() int HitIndex() const { return mId >> 8; }

 private:
  int mId;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCHITID_H
