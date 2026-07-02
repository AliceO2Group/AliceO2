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

/// \file ClusterAccumulator.cxx
/// \author Felix Weiglhofer

#include "ClusterAccumulator.h"
#include "CfUtils.h"
#include "GPUParam.h"
#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/ClusterNative.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

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

GPUd() void ClusterAccumulator::finalize(const CfChargePos& pos, const Charge q, TPCTime timeOffset)
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
}

GPUd() bool ClusterAccumulator::toNative(const CfChargePos& pos, const Charge q, tpc::ClusterNative& cn, const GPUParam& param, const CfArray2D<PackedCharge>& chargeMap)
{
  Pad pad = pos.pad();

  bool isEdgeCluster;
  if (param.rec.tpc.cfEdgeTwoPads) {
    isEdgeCluster = pad < 2 || pad >= GPUTPCGeometry::NPads(pos.row()) - 2; // Geometrical edge check, peak within 2 pads of sector edge
    if (isEdgeCluster) {
      bool leftEdge = (pad < 2);
      if (leftEdge ? (pad == 1 && chargeMap[pos.delta({-1, 0})].unpack() < 1) : (pad == (GPUTPCGeometry::NPads(pos.row()) - 2) && chargeMap[pos.delta({1, 0})].unpack() < 1)) {
        isEdgeCluster = false; // No edge cluster if peak is close to edge but no charge at the edge.
      } else if (leftEdge ? (pad < mPadMean) : (pad > mPadMean)) {
        mPadMean = pad; // Correct to peak position if COG is close to middle of pad than peak
      }
    }
  } else {
    isEdgeCluster = pad == 0 || pad == GPUTPCGeometry::NPads(pos.row()) - 1;
  }

  cn.qTotPacked = CAMath::Float2UIntRn(mQtot);
  if (cn.qTotPacked <= param.rec.tpc.cfQTotCutoff) {
    return false;
  }
  cn.qMax = q; // cfQMaxCutoff check already done at PeakFinder level
  if (mTimeMean < param.rec.tpc.clustersShiftTimebinsClusterizer) {
    return false;
  }
  if (q <= param.rec.tpc.cfQMaxCutoffSingleTime && mTimeSigma == 0) {
    return false;
  }
  if (q <= param.rec.tpc.cfQMaxCutoffSinglePad && mPadSigma == 0) {
    return false;
  }

  bool wasSplitInTime = mSplitInTime >= param.rec.tpc.cfMinSplitNum;
  bool wasSplitInPad = mSplitInPad >= param.rec.tpc.cfMinSplitNum;
  bool isSingleCluster = (mPadSigma == 0) || (mTimeSigma == 0);

  uint8_t flags = 0;
  flags |= (isEdgeCluster) ? tpc::ClusterNative::flagEdge : 0;
  flags |= (wasSplitInTime) ? tpc::ClusterNative::flagSplitTime : 0;
  flags |= (wasSplitInPad) ? tpc::ClusterNative::flagSplitPad : 0;
  flags |= (isSingleCluster) ? tpc::ClusterNative::flagSingle : 0;

  cn.setTimeFlags(mTimeMean - param.rec.tpc.clustersShiftTimebinsClusterizer, flags);
  cn.setPad(mPadMean);
  cn.setSigmaTime(mTimeSigma);
  cn.setSigmaPad(mPadSigma);

  return true;
}
