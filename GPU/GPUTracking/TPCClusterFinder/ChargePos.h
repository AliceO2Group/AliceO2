// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChargePos.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CHARGE_POS_H
#define O2_GPU_CHARGE_POS_H

#include "clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct ChargePos {
  GlobalPad gpad;
  Timestamp timePadded;

  GPUdDefault() ChargePos() CON_DEFAULT;

  GPUdi() ChargePos(Row row, Pad pad, Timestamp t)
  {
    gpad = tpcGlobalPadIdx(row, pad);
    timePadded = t + PADDING_TIME;
  }

  GPUdi() ChargePos(const GlobalPad& p, const Timestamp& t) : gpad(p), timePadded(t) {}

  GPUdi() ChargePos delta(const Delta2& d) const
  {
    return {GlobalPad(gpad + d.x), Timestamp(timePadded + d.y)};
  }

  GPUdi() Row row() const { return gpad / TPC_PADS_PER_ROW_PADDED; }
  GPUdi() Pad pad() const { return gpad % TPC_PADS_PER_ROW_PADDED - PADDING_PAD; }
  GPUdi() Timestamp time() const { return timePadded - PADDING_TIME; }

  GPUdi() bool isPadding() const
  {
    Pad pad = gpad % TPC_PADS_PER_ROW_PADDED;
    return timePadded < PADDING_TIME || timePadded >= TPC_MAX_TIME || pad < PADDING_PAD;
  }

 private:
  // Maps the position of a pad given as row and index in that row to a unique
  // index between 0 and TPC_NUM_OF_PADS.
  static GPUdi() GlobalPad tpcGlobalPadIdx(Row row, Pad pad)
  {
    return TPC_PADS_PER_ROW_PADDED * row + pad + PADDING_PAD;
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
