// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterizer.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CLUSTERIZER_H
#define O2_GPU_CLUSTERIZER_H

#include "clusterFinderDefs.h"
#include "GPUTPCClusterFinderKernels.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class ClusterAccumulator;

namespace deprecated
{
class ClusterNavite;
}

class Clusterizer
{

 public:
  static GPUd() void computeClustersImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const PackedCharge*, GPUglobalref() const deprecated::Digit*, uint, uint, GPUglobalref() uint*, GPUglobalref() deprecated::ClusterNative*);

 private:
  static GPUd() void addOuterCharge(GPUglobalref() const PackedCharge*, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() Charge addInnerCharge(GPUglobalref() const PackedCharge*, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void addCorner(GPUglobalref() const PackedCharge*, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void addLine(GPUglobalref() const PackedCharge*, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void updateClusterScratchpadInner(ushort, ushort, GPUsharedref() const PackedCharge*, ClusterAccumulator*, GPUsharedref() uchar*);

  static GPUd() void updateClusterScratchpadOuter(ushort, ushort, ushort, ushort, GPUsharedref() const PackedCharge*, ClusterAccumulator*);

  static GPUd() void buildClusterScratchPad(GPUglobalref() const PackedCharge*, ChargePos, GPUsharedref() ChargePos*, GPUsharedref() PackedCharge*, GPUsharedref() uchar*, ClusterAccumulator*);

  static GPUd() void buildClusterNaive(GPUglobalref() const PackedCharge*, ClusterAccumulator*, GlobalPad, Timestamp);

  static GPUd() void sortIntoBuckets(const deprecated::ClusterNative*, const uint, const uint, GPUglobalref() uint*, GPUglobalref() deprecated::ClusterNative*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
