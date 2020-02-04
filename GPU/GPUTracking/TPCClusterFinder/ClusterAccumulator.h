// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterAccumulator.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CLUSTER_ACCUMULATOR_H
#define O2_GPU_CLUSTER_ACCUMULATOR_H

#include "clusterFinderDefs.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{

namespace tpc
{
struct ClusterNative;
}

namespace gpu
{

class ClusterAccumulator
{

 public:
  GPUd() Charge updateInner(PackedCharge, Delta2);
  GPUd() Charge updateOuter(PackedCharge, Delta2);

  GPUd() void finalize(const deprecated::Digit&);
  GPUd() void toNative(const deprecated::Digit&, tpc::ClusterNative&) const;

 private:
  float mQtot = 0;
  float mPadMean = 0;
  float mPadSigma = 0;
  float mTimeMean = 0;
  float mTimeSigma = 0;
  uchar mSplitInTime = 0;
  uchar mSplitInPad = 0;

  GPUd() void update(Charge, Delta2);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
