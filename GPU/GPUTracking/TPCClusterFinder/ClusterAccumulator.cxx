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

#if !defined(__OPENCL__)
#include <cmath>
#endif

using namespace std;

namespace GPUCA_NAMESPACE
{
namespace gpu
{

using namespace deprecated;

GPUd() void reset(ClusterAccumulator* clus)
{
  clus->Q = 0.f;
  clus->padMean = 0.f;
  clus->timeMean = 0.f;
  clus->padSigma = 0.f;
  clus->timeSigma = 0.f;
  clus->splitInTime = 0;
  clus->splitInPad = 0;
}

GPUd() void toNative(const ClusterAccumulator* cluster, const Digit* d, ClusterNative* cn)
{
  uchar isEdgeCluster = CfUtils::isAtEdge(d);
  uchar splitInTime = cluster->splitInTime >= MIN_SPLIT_NUM;
  uchar splitInPad = cluster->splitInPad >= MIN_SPLIT_NUM;
  uchar flags =
    (isEdgeCluster << CN_FLAG_POS_IS_EDGE_CLUSTER) | (splitInTime << CN_FLAG_POS_SPLIT_IN_TIME) | (splitInPad << CN_FLAG_POS_SPLIT_IN_PAD);

  cn->qmax = d->charge;
  cn->qtot = cluster->Q;
  cnSetTimeFlags(cn, cluster->timeMean, flags);
  cnSetPad(cn, cluster->padMean);
  cnSetSigmaTime(cn, cluster->timeSigma);
  cnSetSigmaPad(cn, cluster->padSigma);
}

GPUd() void collectCharge(
  ClusterAccumulator* cluster,
  Charge splitCharge,
  Delta dp,
  Delta dt)
{
  cluster->Q += splitCharge;
  cluster->padMean += splitCharge * dp;
  cluster->timeMean += splitCharge * dt;
  cluster->padSigma += splitCharge * dp * dp;
  cluster->timeSigma += splitCharge * dt * dt;
}

GPUd() Charge updateClusterInner(
  ClusterAccumulator* cluster,
  PackedCharge charge,
  Delta dp,
  Delta dt)
{
  Charge q = charge.unpack();

  collectCharge(cluster, q, dp, dt);

  bool split = charge.isSplit();
  cluster->splitInTime += (dt != 0 && split);
  cluster->splitInPad += (dp != 0 && split);

  return q;
}

GPUd() void updateClusterOuter(
  ClusterAccumulator* cluster,
  PackedCharge charge,
  Delta dp,
  Delta dt)
{
  Charge q = charge.unpack();

  bool split = charge.isSplit();
  bool has3x3 = charge.has3x3Peak();

  collectCharge(cluster, (has3x3) ? 0.f : q, dp, dt);

  cluster->splitInTime += (dt != 0 && split && !has3x3);
  cluster->splitInPad += (dp != 0 && split && !has3x3);
}

GPUd() void finalize(
  ClusterAccumulator* pc,
  const Digit* myDigit)
{
  pc->Q += myDigit->charge;
  if (pc->Q == 0) {
    return; // TODO: Why does this happen?
  }

  pc->padMean /= pc->Q;
  pc->timeMean /= pc->Q;
  pc->padSigma /= pc->Q;
  pc->timeSigma /= pc->Q;

  pc->padSigma = sqrt(pc->padSigma - pc->padMean * pc->padMean);
  pc->timeSigma = sqrt(pc->timeSigma - pc->timeMean * pc->timeMean);

  pc->padMean += myDigit->pad;
  pc->timeMean += myDigit->time;

#if defined(CORRECT_EDGE_CLUSTERS)
  if (CfUtils::isAtEdge(myDigit)) {
    float s = (myDigit->pad < 2) ? 1.f : -1.f;
    bool c = s * (pc->padMean - myDigit->pad) > 0.f;
    pc->padMean = (c) ? myDigit->pad : pc->padMean;
  }
#endif
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
