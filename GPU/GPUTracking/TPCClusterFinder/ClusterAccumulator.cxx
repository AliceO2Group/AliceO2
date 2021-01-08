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

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

GPUd() void ClusterAccumulator::toNative(const ChargePos& pos, Charge q, int minSplitNum, tpc::ClusterNative& cn) const
{
  bool isEdgeCluster = CfUtils::isAtEdge(pos);
  bool wasSplitInTime = mSplitInTime >= minSplitNum;
  bool wasSplitInPad = mSplitInPad >= minSplitNum;
  bool isSingleCluster = (mPadSigma == 0) || (mTimeSigma == 0);

  uchar flags = 0;
  flags |= (isEdgeCluster) ? tpc::ClusterNative::flagEdge : 0;
  flags |= (wasSplitInTime) ? tpc::ClusterNative::flagSplitTime : 0;
  flags |= (wasSplitInPad) ? tpc::ClusterNative::flagSplitPad : 0;
  flags |= (isSingleCluster) ? tpc::ClusterNative::flagSingle : 0;

  cn.qMax = q;
  cn.qTot = mQtot;
  cn.setTimeFlags(mTimeMean, flags);
  cn.setPad(mPadMean);
  cn.setSigmaTime(mTimeSigma);
  cn.setSigmaPad(mPadSigma);
}

GPUd() void ClusterAccumulator::update(Charge splitCharge, Delta2 d)
{
  mQtot += splitCharge;
  mPadMean += splitCharge * d.x;
  mTimeMean += splitCharge * d.y;
  mPadSigma += splitCharge * d.x * d.x;
  mTimeSigma += splitCharge * d.y * d.y;
}

GPUd() Charge ClusterAccumulator::updateInner(PackedCharge charge, Delta2 d)
{
  Charge q = charge.unpack();

  update(q, d);

  bool split = charge.isSplit();
  mSplitInTime += (d.y != 0 && split);
  mSplitInPad += (d.x != 0 && split);

  return q;
}

GPUd() Charge ClusterAccumulator::updateOuter(PackedCharge charge, Delta2 d)
{
  Charge q = charge.unpack();

  bool split = charge.isSplit();
  bool has3x3 = charge.has3x3Peak();

  update((has3x3) ? 0.f : q, d);

  mSplitInTime += (d.y != 0 && split && !has3x3);
  mSplitInPad += (d.x != 0 && split && !has3x3);

  return q;
}

GPUd() void ClusterAccumulator::finalize(const ChargePos& pos, Charge q, TPCTime timeOffset)
{
  mQtot += q;

  mPadMean /= mQtot;
  mTimeMean /= mQtot;
  mPadSigma /= mQtot;
  mTimeSigma /= mQtot;

  mPadSigma = CAMath::Sqrt(mPadSigma - mPadMean * mPadMean);
  mTimeSigma = CAMath::Sqrt(mTimeSigma - mTimeMean * mTimeMean);

  Pad pad = pos.pad();
  mPadMean += pad;
  mTimeMean += timeOffset + pos.time();

  if (CfUtils::isAtEdge(pos)) {
    bool leftEdge = (pad < 2);
    bool correct = (leftEdge) ? (pad < mPadMean) : (pad > mPadMean);
    mPadMean = (correct) ? pad : mPadMean;
  }
}
