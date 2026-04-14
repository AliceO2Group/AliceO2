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

/// \file CfChargePos.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CHARGE_POS_H
#define O2_GPU_CHARGE_POS_H

#include "clusterFinderDefs.h"

namespace o2::gpu
{

#define INVALID_TIME_BIN (-GPUCF_PADDING_TIME - 1)

struct CfChargePos {
  tpccf::GlobalPad gpad;
  tpccf::TPCFragmentTime timePadded;

  GPUdDefault() CfChargePos() = default;

  constexpr GPUhdi() CfChargePos(tpccf::Row row, tpccf::Pad pad, tpccf::TPCFragmentTime t)
    : gpad(tpcGlobalPadIdx(row, pad)), timePadded(t + GPUCF_PADDING_TIME)
  {
  }

  GPUdi() CfChargePos(const tpccf::GlobalPad& p, const tpccf::TPCFragmentTime& t) : gpad(p), timePadded(t) {}

  GPUdi() CfChargePos delta(const tpccf::Delta2& d) const
  {
    return {tpccf::GlobalPad(gpad + d.x), tpccf::TPCFragmentTime(timePadded + d.y)};
  }

  GPUdi() bool valid() const { return timePadded >= 0; }

  GPUdi() tpccf::Row row() const { return gpad / TPC_CLUSTERER_ROW_STRIDE; }
  GPUdi() tpccf::Pad pad() const { return gpad % TPC_CLUSTERER_ROW_STRIDE - GPUCF_PADDING_PAD; }
  GPUdi() tpccf::TPCFragmentTime time() const { return timePadded - GPUCF_PADDING_TIME; }
  GPUdi() tpccf::TPCFragmentTime globalTime() const { return timePadded; }

 private:
  // Maps the position of a pad given as row and index in that row to a unique
  // index between 0 and TPC_NUM_OF_PADS.
  static constexpr GPUdi() tpccf::GlobalPad tpcGlobalPadIdx(tpccf::Row row, tpccf::Pad pad)
  {
    return TPC_CLUSTERER_ROW_STRIDE * row + pad + GPUCF_PADDING_PAD;
  }
};

inline constexpr CfChargePos INVALID_CHARGE_POS{255, 255, INVALID_TIME_BIN};

} // namespace o2::gpu

#endif
