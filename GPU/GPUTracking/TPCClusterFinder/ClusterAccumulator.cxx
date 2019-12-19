// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterAccumulator.cxx
/// \author Felix Weiglhofer

#include "ClusterAccumulator.h"
#include "CfUtils.h"
#include "ClusterNative.h"

#if !defined(__OPENCL__)
#include <cmath>
#endif

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

GPUd() void ClusterAccumulator::toNative(const deprecated::Digit& d, deprecated::ClusterNative& cn) const
{
  uchar isEdgeCluster = CfUtils::isAtEdge(&d);
  uchar wasSplitInTime = splitInTime >= MIN_SPLIT_NUM;
  uchar wasSplitInPad = splitInPad >= MIN_SPLIT_NUM;
  uchar flags =
    (isEdgeCluster << deprecated::CN_FLAG_POS_IS_EDGE_CLUSTER) | (wasSplitInTime << deprecated::CN_FLAG_POS_SPLIT_IN_TIME) | (wasSplitInPad << deprecated::CN_FLAG_POS_SPLIT_IN_PAD);

  cn.qmax = d.charge;
  cn.qtot = qtot;
  deprecated::cnSetTimeFlags(&cn, timeMean, flags);
  deprecated::cnSetPad(&cn, padMean);
  deprecated::cnSetSigmaTime(&cn, timeSigma);
  deprecated::cnSetSigmaPad(&cn, padSigma);
}

GPUd() void ClusterAccumulator::update(Charge splitCharge, Delta dp, Delta dt)
{
  qtot += splitCharge;
  padMean += splitCharge * dp;
  timeMean += splitCharge * dt;
  padSigma += splitCharge * dp * dp;
  timeSigma += splitCharge * dt * dt;
}

GPUd() Charge ClusterAccumulator::updateInner(PackedCharge charge, Delta dp, Delta dt)
{
  Charge q = charge.unpack();

  update(q, dp, dt);

  bool split = charge.isSplit();
  splitInTime += (dt != 0 && split);
  splitInPad += (dp != 0 && split);

  return q;
}

GPUd() Charge ClusterAccumulator::updateOuter(PackedCharge charge, Delta dp, Delta dt)
{
  Charge q = charge.unpack();

  bool split = charge.isSplit();
  bool has3x3 = charge.has3x3Peak();

  update((has3x3) ? 0.f : q, dp, dt);

  splitInTime += (dt != 0 && split && !has3x3);
  splitInPad += (dp != 0 && split && !has3x3);

  return q;
}

GPUd() void ClusterAccumulator::finalize(const deprecated::Digit& myDigit)
{
  qtot += myDigit.charge;
  if (qtot == 0) {
    return; // TODO: Why does this happen?
  }

  padMean /= qtot;
  timeMean /= qtot;
  padSigma /= qtot;
  timeSigma /= qtot;

  padSigma = sqrt(padSigma - padMean * padMean);
  timeSigma = sqrt(timeSigma - timeMean * timeMean);

  padMean += myDigit.pad;
  timeMean += myDigit.time;

#if defined(CORRECT_EDGE_CLUSTERS)
  if (CfUtils::isAtEdge(myDigit)) {
    float s = (myDigit->pad < 2) ? 1.f : -1.f;
    bool c = s * (padMean - myDigit->pad) > 0.f;
    padMean = (c) ? myDigit->pad : padMean;
  }
#endif
}
