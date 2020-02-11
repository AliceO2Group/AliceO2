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

GPUd() void ClusterAccumulator::toNative(const deprecated::Digit& d, tpc::ClusterNative& cn) const
{
  bool isEdgeCluster = CfUtils::isAtEdge(&d);
  bool wasSplitInTime = mSplitInTime >= MIN_SPLIT_NUM;
  bool wasSplitInPad = mSplitInPad >= MIN_SPLIT_NUM;

  uchar flags = 0;
  flags |= (isEdgeCluster) ? tpc::ClusterNative::flagEdge : 0;
  flags |= (wasSplitInTime) ? tpc::ClusterNative::flagSplitTime : 0;
  flags |= (wasSplitInPad) ? tpc::ClusterNative::flagSplitPad : 0;

  cn.qMax = d.charge;
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

GPUd() void ClusterAccumulator::finalize(const deprecated::Digit& myDigit)
{
  mQtot += myDigit.charge;
  if (mQtot == 0) {
    return; // TODO: Why does this happen?
  }

  mPadMean /= mQtot;
  mTimeMean /= mQtot;
  mPadSigma /= mQtot;
  mTimeSigma /= mQtot;

  mPadSigma = CAMath::Sqrt(mPadSigma - mPadMean * mPadMean);
  mTimeSigma = CAMath::Sqrt(mTimeSigma - mTimeMean * mTimeMean);

  mPadMean += myDigit.pad;
  mTimeMean += myDigit.time;

#if defined(CORRECT_EDGE_CLUSTERS)
  if (CfUtils::isAtEdge(myDigit)) {
    float s = (myDigit->pad < 2) ? 1.f : -1.f;
    bool c = s * (mPadMean - myDigit->pad) > 0.f;
    mPadMean = (c) ? myDigit->pad : mPadMean;
  }
#endif
}
