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
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{

namespace tpc
{
class ClusterNative;
}

namespace gpu
{

class ClusterAccumulator;

class Clusterizer
{

 public:
  static GPUd() void computeClustersImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, const Array2D<PackedCharge>&, GPUglobalref() const deprecated::Digit*, uint, uint, GPUglobalref() uint*, GPUglobalref() tpc::ClusterNative*);

 private:
  static GPUd() void addOuterCharge(const Array2D<PackedCharge>&, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() Charge addInnerCharge(const Array2D<PackedCharge>&, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void addCorner(const Array2D<PackedCharge>&, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void addLine(const Array2D<PackedCharge>&, ClusterAccumulator*, GlobalPad, Timestamp, Delta, Delta);

  static GPUd() void updateClusterScratchpadInner(ushort, ushort, GPUsharedref() const PackedCharge*, ClusterAccumulator*, GPUsharedref() uchar*);

  static GPUd() void updateClusterScratchpadOuter(ushort, ushort, ushort, ushort, GPUsharedref() const PackedCharge*, ClusterAccumulator*);

  static GPUd() void buildClusterScratchPad(const Array2D<PackedCharge>&, ChargePos, GPUsharedref() ChargePos*, GPUsharedref() PackedCharge*, GPUsharedref() uchar*, ClusterAccumulator*);

  static GPUd() void buildClusterNaive(const Array2D<PackedCharge>&, ClusterAccumulator*, GlobalPad, Timestamp);

  static GPUd() void sortIntoBuckets(const tpc::ClusterNative&, const uint, const uint, GPUglobalref() uint*, GPUglobalref() tpc::ClusterNative*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
