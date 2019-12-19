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
namespace gpu
{

class ClusterAccumulator
{

 public:
  GPUd() Charge updateInner(PackedCharge, Delta, Delta);
  GPUd() Charge updateOuter(PackedCharge, Delta, Delta);

  GPUd() void finalize(const deprecated::Digit&);
  GPUd() void toNative(const deprecated::Digit&, deprecated::ClusterNative&) const;

 private:
  float qtot = 0;
  float padMean = 0;
  float padSigma = 0;
  float timeMean = 0;
  float timeSigma = 0;
  uchar splitInTime = 0;
  uchar splitInPad = 0;

  GPUd() void update(Charge, Delta, Delta);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
