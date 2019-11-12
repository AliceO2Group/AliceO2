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

#include "GPUDef.h"
#include "PackedCharge.h"
#include "cl/clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#if 0
class ClusterAccumulator
{

 public:
  GPUd() float updateInner(float, delta_t, delta_t);
  GPUd() float updateOuter(float, delta_t, delta_t);

  GPUd() void toNative(const Digit&, ClusterNative&);

 private:
  float qtot = 0;
  float padMean = 0;
  float padSigma = 0;
  float timeMean = 0;
  float timeSigma = 0;
  uchar splitInTime = 0;
  uchar splitInPad = 0;

  GPUd() void update(float, delta_t, delta_t);
  GPUd() void finalize(const Digit&);
};
#endif

struct ClusterAccumulator {
  Charge Q;
  Charge padMean;
  Charge padSigma;
  Charge timeMean;
  Charge timeSigma;
  uchar splitInTime;
  uchar splitInPad;
};

GPUd() void toNative(const ClusterAccumulator*, const deprecated::Digit*, deprecated::ClusterNative*);
GPUd() Charge updateClusterInner(ClusterAccumulator*, PackedCharge, Delta, Delta);
GPUd() void updateClusterOuter(ClusterAccumulator*, PackedCharge, Delta, Delta);
GPUd() void reset(ClusterAccumulator*);
GPUd() void finalize(ClusterAccumulator*, const deprecated::Digit*);

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
